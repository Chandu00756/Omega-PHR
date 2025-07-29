"""
Protocol Buffer code generation utilities for Ω-PHR framework.

Automates the generation of Python stubs from .proto files with
proper import handling and directory structure management.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


class ProtoCodeGen:
    """Advanced Protocol Buffer code generation with import optimization."""

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize code generator.

        Args:
            project_root: Project root directory (auto-detected if None)
        """
        self.project_root = project_root or self._detect_project_root()
        self.proto_dirs = []
        self.output_dirs = []

    def _detect_project_root(self) -> Path:
        """Auto-detect project root directory."""
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
                return current
            current = current.parent
        return Path.cwd()

    def add_proto_directory(
        self, proto_dir: Path, output_dir: Optional[Path] = None
    ) -> None:
        """
        Add a directory containing .proto files.

        Args:
            proto_dir: Directory containing .proto files
            output_dir: Output directory for generated files (defaults to proto_dir)
        """
        proto_dir = Path(proto_dir)
        output_dir = output_dir or proto_dir

        if not proto_dir.exists():
            raise ValueError(f"Proto directory does not exist: {proto_dir}")

        self.proto_dirs.append(proto_dir)
        self.output_dirs.append(Path(output_dir))

    def discover_proto_files(self) -> List[tuple[Path, Path]]:
        """
        Discover all .proto files in registered directories.

        Returns:
            List of (proto_file, output_dir) tuples
        """
        proto_files = []

        for proto_dir, output_dir in zip(self.proto_dirs, self.output_dirs):
            for proto_file in proto_dir.rglob("*.proto"):
                proto_files.append((proto_file, output_dir))

        return proto_files

    def generate_code(
        self,
        proto_files: Optional[List[Path]] = None,
        include_dirs: Optional[List[Path]] = None,
        grpc_python_out: bool = True,
        python_out: bool = True,
        mypy_out: bool = False,
        experimental_allow_proto3_optional: bool = True,
    ) -> bool:
        """
        Generate Python code from Protocol Buffer files.

        Args:
            proto_files: Specific proto files to compile (None for all discovered)
            include_dirs: Additional include directories
            grpc_python_out: Generate gRPC Python stubs
            python_out: Generate Python message stubs
            mypy_out: Generate mypy stubs
            experimental_allow_proto3_optional: Allow proto3 optional fields

        Returns:
            True if generation succeeded
        """
        if proto_files is None:
            proto_file_pairs = self.discover_proto_files()
        else:
            proto_file_pairs = [(f, self.output_dirs[0]) for f in proto_files]

        if not proto_file_pairs:
            print("No .proto files found")
            return True

        # Prepare include directories
        include_dirs = include_dirs or []
        all_includes = list(include_dirs)
        all_includes.extend(self.proto_dirs)

        # Add common proto includes
        all_includes.extend(self._get_well_known_proto_includes())

        success = True

        for proto_file, output_dir in proto_file_pairs:
            if not self._compile_proto_file(
                proto_file,
                output_dir,
                all_includes,
                grpc_python_out,
                python_out,
                mypy_out,
                experimental_allow_proto3_optional,
            ):
                success = False

        if success:
            self._post_process_generated_files()

        return success

    def _compile_proto_file(
        self,
        proto_file: Path,
        output_dir: Path,
        include_dirs: List[Path],
        grpc_python_out: bool,
        python_out: bool,
        mypy_out: bool,
        experimental_allow_proto3_optional: bool,
    ) -> bool:
        """Compile a single proto file."""
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build protoc command
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={proto_file.parent}",
        ]

        # Add include directories
        for include_dir in include_dirs:
            if include_dir != proto_file.parent:  # Avoid duplicates
                cmd.append(f"--proto_path={include_dir}")

        # Add output options
        if python_out:
            cmd.append(f"--python_out={output_dir}")

        if grpc_python_out:
            cmd.append(f"--grpc_python_out={output_dir}")

        if mypy_out:
            cmd.append(f"--mypy_out={output_dir}")

        # Add experimental options
        if experimental_allow_proto3_optional:
            cmd.append("--experimental_allow_proto3_optional")

        # Add proto file
        cmd.append(str(proto_file))

        print(f"Compiling {proto_file.name}...")

        try:
            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                print(f"Error compiling {proto_file}:")
                print(result.stderr)
                return False

            if result.stdout.strip():
                print(result.stdout)

            print(f"✓ Generated stubs for {proto_file.name}")
            return True

        except Exception as e:
            print(f"Exception compiling {proto_file}: {e}")
            return False

    def _get_well_known_proto_includes(self) -> List[Path]:
        """Get paths to well-known Protocol Buffer includes."""
        includes = []

        # Try to find grpc_tools includes
        try:
            import grpc_tools

            grpc_tools_path = Path(grpc_tools.__file__).parent
            includes.append(grpc_tools_path / "_proto")
        except ImportError:
            pass

        # Try to find protobuf includes
        try:
            import google.protobuf

            protobuf_path = Path(google.protobuf.__file__).parent.parent
            includes.append(protobuf_path)
        except ImportError:
            pass

        return [p for p in includes if p.exists()]

    def _post_process_generated_files(self) -> None:
        """Post-process generated files to fix imports and add type hints."""
        for output_dir in self.output_dirs:
            self._fix_relative_imports(output_dir)
            self._add_init_files(output_dir)

    def _fix_relative_imports(self, output_dir: Path) -> None:
        """Fix relative imports in generated Python files."""
        for py_file in output_dir.rglob("*_pb2*.py"):
            if py_file.name in ("__init__.py",):
                continue

            try:
                content = py_file.read_text()
                original_content = content

                # Fix common import issues
                import_fixes = [
                    # Fix protobuf imports
                    ("from google.protobuf", "from google.protobuf"),
                    # Fix grpc imports
                    ("import grpc", "import grpc"),
                    # Fix relative proto imports
                    ("import (.+)_pb2", r"from . import \1_pb2"),
                ]

                for pattern, replacement in import_fixes:
                    import re

                    content = re.sub(pattern, replacement, content)

                # Only write if content changed
                if content != original_content:
                    py_file.write_text(content)
                    print(f"✓ Fixed imports in {py_file.name}")

            except Exception as e:
                print(f"Warning: Could not fix imports in {py_file}: {e}")

    def _add_init_files(self, output_dir: Path) -> None:
        """Add __init__.py files to make packages importable."""
        # Add __init__.py to output directory if it doesn't exist
        init_file = output_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Generated Protocol Buffer stubs."""\n')

        # Add __init__.py to subdirectories containing generated files
        for subdir in output_dir.rglob("*"):
            if subdir.is_dir() and any(subdir.glob("*_pb2*.py")):
                sub_init = subdir / "__init__.py"
                if not sub_init.exists():
                    sub_init.write_text('"""Generated Protocol Buffer stubs."""\n')

    def clean_generated_files(self) -> None:
        """Clean all generated files."""
        patterns = ["*_pb2.py", "*_pb2_grpc.py", "*_pb2.pyi", "*_pb2_grpc.pyi"]

        for output_dir in self.output_dirs:
            for pattern in patterns:
                for generated_file in output_dir.rglob(pattern):
                    try:
                        generated_file.unlink()
                        print(f"✓ Removed {generated_file}")
                    except Exception as e:
                        print(f"Warning: Could not remove {generated_file}: {e}")


def generate_all_protos(project_root: Optional[Path] = None) -> bool:
    """
    Generate all Protocol Buffer stubs in the project.

    Args:
        project_root: Project root directory

    Returns:
        True if generation succeeded
    """
    codegen = ProtoCodeGen(project_root)

    # Auto-discover proto directories
    if project_root:
        root = Path(project_root)
    else:
        root = codegen.project_root

    # Add common proto directories
    for proto_dir in root.rglob("proto"):
        if proto_dir.is_dir() and any(proto_dir.glob("*.proto")):
            codegen.add_proto_directory(proto_dir)

    # Generate code
    return codegen.generate_code()


def main():
    """Command-line interface for proto code generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Protocol Buffer stubs")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    parser.add_argument("--clean", action="store_true", help="Clean generated files")
    parser.add_argument(
        "--proto-dir", type=Path, action="append", help="Proto directories"
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory")

    args = parser.parse_args()

    codegen = ProtoCodeGen(args.project_root)

    if args.proto_dir:
        for proto_dir in args.proto_dir:
            codegen.add_proto_directory(proto_dir, args.output_dir)
    else:
        # Auto-discover
        root = args.project_root or codegen.project_root
        for proto_dir in root.rglob("proto"):
            if proto_dir.is_dir() and any(proto_dir.glob("*.proto")):
                codegen.add_proto_directory(proto_dir)

    if args.clean:
        codegen.clean_generated_files()
    else:
        success = codegen.generate_code()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
