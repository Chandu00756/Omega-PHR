"""
Timeline Lattice Repository - Secure Temporal Data Management

Advanced repository implementation for storing and retrieving temporal events
with research-grade stability, cryptographic verification, and paradox containment.
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.cluster import Cluster
    from cassandra.policies import DCAwareRoundRobinPolicy
    from cassandra.query import PreparedStatement, SimpleStatement

    CASSANDRA_AVAILABLE = True
except ImportError:
    CASSANDRA_AVAILABLE = False

from .config import DatabaseType, TimelineServiceConfig
from .models import ParadoxType, TemporalParadox, Timeline, TimelineEvent

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base class for timeline repositories with common interface."""

    async def store_event(self, event: TimelineEvent) -> bool:
        """Store a timeline event with cryptographic verification."""
        raise NotImplementedError

    async def get_events(
        self,
        timeline_id: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> list[TimelineEvent]:
        """Retrieve events from a timeline with temporal filtering."""
        raise NotImplementedError

    async def create_timeline(self, timeline: Timeline) -> bool:
        """Create a new timeline with metadata."""
        raise NotImplementedError

    async def branch_timeline(
        self, source_id: str, branch_point: int, new_timeline_id: str
    ) -> bool:
        """Create a branched timeline from a specific point in time."""
        raise NotImplementedError

    async def detect_paradoxes(self, timeline_id: str) -> list[TemporalParadox]:
        """Analyze timeline for temporal paradoxes and inconsistencies."""
        raise NotImplementedError


class ScyllaRepository(BaseRepository):
    """ScyllaDB-based repository for timeline data."""

    def __init__(self, config: TimelineServiceConfig):
        self.config = config
        self.cluster: Any | None = None
        self.session: Any | None = None
        self._prepared_statements: dict[str, Any] = {}

    def _ensure_connection(self) -> bool:
        """Ensure connection (session) is available, return False if not."""
        if self.session is None:
            logger.error("ScyllaDB session not available")
            return False
        return True

    def _ensure_session(self) -> bool:
        """Ensure session is available, return False if not."""
        if self.session is None:
            logger.error("ScyllaDB session not available")
            return False
        return True

    async def initialize(self):
        """Initialize ScyllaDB connection and prepare statements."""
        if not CASSANDRA_AVAILABLE:
            raise RuntimeError("Cassandra driver not available")

        # Configure authentication if provided
        auth_provider = None
        if self.config.database.username and self.config.database.password:
            auth_provider = PlainTextAuthProvider(
                username=self.config.database.username,
                password=self.config.database.password,
            )

        # Create cluster connection
        self.cluster = Cluster(
            contact_points=self.config.database.hosts,
            port=self.config.database.port,
            auth_provider=auth_provider,
            load_balancing_policy=DCAwareRoundRobinPolicy(),
            protocol_version=4,
        )

        if self.cluster is not None:
            self.session = self.cluster.connect()
        else:
            logger.error("Cluster not available, session will be None")
            self.session = None

        # Create keyspace if it doesn't exist
        await self._create_keyspace()

        # Set keyspace
        if self.session is not None:
            self.session.set_keyspace(self.config.database.keyspace)
        else:
            logger.warning("Session not available, skipping keyspace selection")

        # Create tables
        await self._create_tables()

        # Prepare statements for better performance
        await self._prepare_statements()

        logger.info(
            f"ScyllaDB repository initialized with keyspace: {self.config.database.keyspace}"
        )

    async def _create_keyspace(self):
        """Create keyspace with proper replication strategy."""
        keyspace_query = f"""
        CREATE KEYSPACE IF NOT EXISTS {self.config.database.keyspace}
        WITH REPLICATION = {{
            'class': 'SimpleStrategy',
            'replication_factor': 1
        }}
        """
        if self.session is not None:
            self.session.execute(keyspace_query)
        else:
            logger.warning("Session not available, skipping keyspace creation")

    async def _create_tables(self):
        """Create all necessary tables for timeline storage."""

        # Timeline events table
        events_table = """
        CREATE TABLE IF NOT EXISTS timeline_events (
            event_id UUID PRIMARY KEY,
            timeline_id TEXT,
            actor_id TEXT,
            parent_id TEXT,
            event_type TEXT,
            payload TEXT,
            metadata MAP<TEXT, TEXT>,
            valid_at_us BIGINT,
            recorded_at_us BIGINT,
            signature BLOB,
            checksum TEXT,
            INDEX(timeline_id),
            INDEX(valid_at_us)
        )
        """

        # Timeline metadata table
        timelines_table = """
        CREATE TABLE IF NOT EXISTS timelines (
            timeline_id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            created_at TIMESTAMP,
            status TEXT,
            metadata MAP<TEXT, TEXT>,
            event_count COUNTER,
            last_modified TIMESTAMP
        )
        """

        # Paradoxes detection table
        paradoxes_table = """
        CREATE TABLE IF NOT EXISTS temporal_paradoxes (
            paradox_id UUID PRIMARY KEY,
            timeline_id TEXT,
            paradox_type TEXT,
            severity FLOAT,
            description TEXT,
            affected_events LIST<UUID>,
            detected_at TIMESTAMP,
            resolved BOOLEAN,
            resolution_method TEXT,
            INDEX(timeline_id),
            INDEX(detected_at)
        )
        """

        # Timeline branches table for tracking forks
        branches_table = """
        CREATE TABLE IF NOT EXISTS timeline_branches (
            branch_id UUID PRIMARY KEY,
            source_timeline_id TEXT,
            target_timeline_id TEXT,
            branch_point_us BIGINT,
            created_at TIMESTAMP,
            INDEX(source_timeline_id)
        )
        """

        for table in [events_table, timelines_table, paradoxes_table, branches_table]:
            if self.session is not None:
                self.session.execute(table)
            else:
                logger.warning("Session not available, skipping table creation")

    async def _prepare_statements(self):
        """Prepare frequently used statements for better performance."""
        if self.session is None:
            logger.warning("Session not available, skipping statement preparation")
            return

        # Event insertion
        self._prepared_statements["insert_event"] = self.session.prepare(
            """
            INSERT INTO timeline_events (
                event_id, timeline_id, actor_id, parent_id, event_type,
                payload, metadata, valid_at_us, recorded_at_us, signature, checksum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        )

        # Event retrieval by timeline
        self._prepared_statements["get_events_by_timeline"] = self.session.prepare(
            """
            SELECT * FROM timeline_events
            WHERE timeline_id = ?
            ORDER BY valid_at_us ASC
        """
        )

        # Timeline creation
        self._prepared_statements["create_timeline"] = self.session.prepare(
            """
            INSERT INTO timelines (
                timeline_id, name, description, created_at, status, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
        """
        )

        # Paradox insertion
        self._prepared_statements["insert_paradox"] = self.session.prepare(
            """
            INSERT INTO temporal_paradoxes (
                paradox_id, timeline_id, paradox_type, severity, description,
                affected_events, detected_at, resolved, resolution_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        )

    async def store_event(self, event: TimelineEvent):
        """Store timeline event."""
        if not self._ensure_connection():
            return

        if self.session is None:
            logger.error("ScyllaDB session not available")
            return

        try:
            if self.session is None:
                logger.error("ScyllaDB session is None, cannot store event")
                return
            if self._prepared_statements.get("insert_event") is None:
                logger.error("Prepared statement for insert_event is not available")
                return
            # Ensure session is not None before executing
            if self.session is not None:
                self.session.execute(
                    self._prepared_statements["insert_event"],
                    [
                        uuid.UUID(event.event_id),
                        event.timeline_id,
                        event.actor_id,
                        event.parent_id,
                        event.event_type.value,
                        event.payload,
                        event.metadata,
                        event.valid_at_us,
                        event.recorded_at_us,
                        event.signature,
                        self._calculate_event_checksum(event),
                    ],
                )
                logger.debug(
                    f"Stored event {event.event_id} for timeline {event.timeline_id}"
                )
            else:
                logger.error("ScyllaDB session became None before storing event")
        except Exception as e:
            logger.error(f"Failed to store event: {e}")

    async def get_events(
        self,
        timeline_id: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> list[TimelineEvent]:
        """Retrieve events with temporal filtering."""
        if self.session is None:
            logger.error("ScyllaDB session not available")
            return []

        try:
            # Base query
            rows = self.session.execute(
                self._prepared_statements["get_events_by_timeline"], [timeline_id]
            )

            events = []
            count = 0

            for row in rows:
                # Apply time filtering
                if start_time and row.valid_at_us < start_time:
                    continue
                if end_time and row.valid_at_us > end_time:
                    continue

                # Apply limit
                if limit and count >= limit:
                    break

                # Verify checksum
                event = TimelineEvent.from_cassandra_row(row)
                if self._verify_event_integrity(event, row.checksum):
                    events.append(event)
                    count += 1
                else:
                    logger.warning(f"Checksum mismatch for event {row.event_id}")

            return events

        except Exception as e:
            logger.error(f"Failed to retrieve events for timeline {timeline_id}: {e}")
            return []

    async def create_timeline(self, timeline: Timeline) -> bool:
        """Create new timeline with metadata."""
        if not self._ensure_session():
            return False

        try:
            self.session.execute(
                self._prepared_statements["create_timeline"],
                [
                    timeline.timeline_id,
                    timeline.name,
                    timeline.description,
                    datetime.now(UTC),
                    timeline.status.value,
                    timeline.metadata,
                ],
            )

            logger.info(f"Created timeline {timeline.timeline_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create timeline {timeline.timeline_id}: {e}")
            return False

    async def branch_timeline(
        self, source_id: str, branch_point: int, new_timeline_id: str
    ) -> bool:
        """Create branched timeline from specific point."""
        if not self._ensure_session():
            return False

        try:
            # Insert branch record
            if self.session is None:
                logger.error("Session not available for branch preparation")
                return False

            branch_stmt = self.session.prepare(
                """
                INSERT INTO timeline_branches (
                    branch_id, source_timeline_id, target_timeline_id,
                    branch_point_us, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """
            )

            self.session.execute(
                branch_stmt,
                [
                    uuid.uuid4(),
                    source_id,
                    new_timeline_id,
                    branch_point,
                    datetime.now(UTC),
                ],
            )

            # Copy events up to branch point
            source_events = await self.get_events(source_id, end_time=branch_point)

            for event in source_events:
                new_event = replace(
                    event, timeline_id=new_timeline_id, event_id=str(uuid.uuid4())
                )
                await self.store_event(new_event)

            logger.info(
                f"Branched timeline {source_id} to {new_timeline_id} at {branch_point}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to branch timeline {source_id}: {e}")
            return False

    async def detect_paradoxes(self, timeline_id: str) -> list[TemporalParadox]:
        """Detect temporal paradoxes in timeline."""
        try:
            events = await self.get_events(timeline_id)
            paradoxes = []

            # Check for causality violations
            for i, event in enumerate(events):
                for _j, other_event in enumerate(events[i + 1 :], i + 1):
                    if self._is_causality_violation(event, other_event):
                        paradox = TemporalParadox(
                            paradox_id=str(uuid.uuid4()),
                            timeline_id=timeline_id,
                            paradox_type=ParadoxType.CAUSALITY_VIOLATION,
                            severity=0.8,
                            description=f"Causality violation between events {event.event_id} and {other_event.event_id}",
                            affected_events=[event.event_id, other_event.event_id],
                            detected_at=datetime.now(UTC),
                        )
                        paradoxes.append(paradox)

                        # Store paradox
                        await self._store_paradox(paradox)

            return paradoxes

        except Exception as e:
            logger.error(f"Failed to detect paradoxes in timeline {timeline_id}: {e}")
            return []

    async def _store_paradox(self, paradox: TemporalParadox):
        """Store detected paradox."""
        if not self._ensure_session():
            return

        if self.session is None:
            logger.error("Session not available for paradox storage")
            return

        self.session.execute(
            self._prepared_statements["insert_paradox"],
            [
                uuid.UUID(paradox.paradox_id),
                paradox.timeline_id,
                paradox.paradox_type.value,
                paradox.severity,
                paradox.description,
                [uuid.UUID(eid) for eid in paradox.affected_events],
                paradox.detected_at,
                paradox.resolved,
                paradox.resolution_method,
            ],
        )

    def _calculate_event_checksum(self, event: TimelineEvent) -> str:
        """Calculate cryptographic checksum for event integrity."""
        content = (
            f"{event.event_id}{event.timeline_id}{event.payload}{event.valid_at_us}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def _verify_event_integrity(
        self, event: TimelineEvent, stored_checksum: str
    ) -> bool:
        """Verify event integrity using checksum."""
        calculated_checksum = self._calculate_event_checksum(event)
        return calculated_checksum == stored_checksum

    def _is_causality_violation(
        self, event1: TimelineEvent, event2: TimelineEvent
    ) -> bool:
        """Check if two events violate causality constraints."""
        # Simple causality check: parent event should come before child
        if (
            event1.event_id == event2.parent_id
            and event1.valid_at_us > event2.valid_at_us
        ):
            return True
        return bool(
            event2.event_id == event1.parent_id
            and event2.valid_at_us > event1.valid_at_us
        )

    async def close(self):
        """Close database connections."""
        if self.cluster:
            self.cluster.shutdown()


class SQLiteRepository(BaseRepository):
    """SQLite-based repository for development and testing."""

    def __init__(self, config: TimelineServiceConfig):
        self.config = config
        self.db_path = Path(config.database.sqlite_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        self._lock = asyncio.Lock()

    def _ensure_connection(self) -> bool:
        """Ensure connection is available, return False if not."""
        if self._connection is None:
            logger.error("SQLite connection not available")
            return False
        return True

    async def initialize(self):
        """Initialize SQLite database and create tables."""
        self._connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit mode
        )
        self._connection.row_factory = sqlite3.Row

        await self._create_tables()
        logger.info(f"SQLite repository initialized at {self.db_path}")

    async def _create_tables(self):
        """Create SQLite tables for timeline storage."""
        if not self._ensure_connection():
            return

        if self._connection is None:
            logger.error("SQLite connection not available for table creation")
            return

        if self._connection is None:
            logger.error("SQLite connection not available")
            return False
        cursor = self._connection.cursor()

        # Timeline events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS timeline_events (
                event_id TEXT PRIMARY KEY,
                timeline_id TEXT NOT NULL,
                actor_id TEXT,
                parent_id TEXT,
                event_type TEXT,
                payload TEXT,
                metadata TEXT,
                valid_at_us INTEGER,
                recorded_at_us INTEGER,
                signature BLOB,
                checksum TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Timeline metadata table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS timelines (
                timeline_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                metadata TEXT,
                event_count INTEGER DEFAULT 0,
                last_modified DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Paradoxes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS temporal_paradoxes (
                paradox_id TEXT PRIMARY KEY,
                timeline_id TEXT,
                paradox_type TEXT,
                severity REAL,
                description TEXT,
                affected_events TEXT,
                detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_method TEXT
            )
        """
        )

        # Timeline branches table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS timeline_branches (
                branch_id TEXT PRIMARY KEY,
                source_timeline_id TEXT,
                target_timeline_id TEXT,
                branch_point_us INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_timeline ON timeline_events(timeline_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_time ON timeline_events(valid_at_us)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_paradoxes_timeline ON temporal_paradoxes(timeline_id)"
        )

        if self._connection is not None:

            self._connection.commit()

    async def store_event(self, event: TimelineEvent) -> bool:
        """Store timeline event in SQLite."""
        if not self._ensure_connection():
            return False

        async with self._lock:
            try:
                checksum = self._calculate_event_checksum(event)

                if not self._ensure_connection():
                    return False

                if self._connection is None:
                    logger.error("SQLite connection not available")
                    return False

                cursor = self._connection.cursor()
                cursor.execute(
                    """
                    INSERT INTO timeline_events (
                        event_id, timeline_id, actor_id, parent_id, event_type,
                        payload, metadata, valid_at_us, recorded_at_us, signature, checksum
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        event.event_id,
                        event.timeline_id,
                        event.actor_id,
                        event.parent_id,
                        event.event_type.value,
                        event.payload,
                        json.dumps(event.metadata) if event.metadata else None,
                        event.valid_at_us,
                        event.recorded_at_us,
                        event.signature,
                        checksum,
                    ],
                )

                if self._connection is not None:

                    self._connection.commit()
                logger.debug(f"Stored event {event.event_id} in SQLite")
                return True

            except Exception as e:
                logger.error(f"Failed to store event in SQLite: {e}")
                return False

    async def get_events(
        self,
        timeline_id: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> list[TimelineEvent]:
        """Retrieve events from SQLite."""
        async with self._lock:
            try:
                if not self._ensure_connection():
                    return []

                if self._connection is None:
                    logger.error("SQLite connection not available")
                    return []

                cursor = self._connection.cursor()

                query = "SELECT * FROM timeline_events WHERE timeline_id = ?"
                params: list[Any] = [timeline_id]

                if start_time:
                    query += " AND valid_at_us >= ?"
                    params.append(start_time)

                if end_time:
                    query += " AND valid_at_us <= ?"
                    params.append(end_time)

                query += " ORDER BY valid_at_us ASC"

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                events = []
                for row in rows:
                    event = TimelineEvent.from_sqlite_row(dict(row))
                    if self._verify_event_integrity(event, row["checksum"]):
                        events.append(event)
                    else:
                        logger.warning(f"Checksum mismatch for event {row['event_id']}")

                return events

            except Exception as e:
                logger.error(f"Failed to retrieve events from SQLite: {e}")
                return []

    async def create_timeline(self, timeline: Timeline) -> bool:
        """Create timeline in SQLite."""
        async with self._lock:
            try:
                if not self._ensure_connection():

                    return False

                cursor = self._connection.cursor()
                cursor.execute(
                    """
                    INSERT INTO timelines (timeline_id, name, description, status, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    [
                        timeline.timeline_id,
                        timeline.name,
                        timeline.description,
                        timeline.status.value,
                        json.dumps(timeline.metadata) if timeline.metadata else None,
                    ],
                )

                if self._connection is not None:

                    self._connection.commit()
                logger.info(f"Created timeline {timeline.timeline_id} in SQLite")
                return True

            except Exception as e:
                logger.error(f"Failed to create timeline in SQLite: {e}")
                return False

    async def branch_timeline(
        self, source_id: str, branch_point: int, new_timeline_id: str
    ) -> bool:
        """Create branched timeline in SQLite."""
        async with self._lock:
            try:
                if not self._ensure_connection():

                    return False

                cursor = self._connection.cursor()

                # Insert branch record
                cursor.execute(
                    """
                    INSERT INTO timeline_branches
                    (branch_id, source_timeline_id, target_timeline_id, branch_point_us)
                    VALUES (?, ?, ?, ?)
                """,
                    [str(uuid.uuid4()), source_id, new_timeline_id, branch_point],
                )

                # Copy events up to branch point
                source_events = await self.get_events(source_id, end_time=branch_point)

                for event in source_events:
                    new_event = replace(
                        event, timeline_id=new_timeline_id, event_id=str(uuid.uuid4())
                    )
                    await self.store_event(new_event)

                if self._connection is not None:

                    self._connection.commit()
                logger.info(f"Branched timeline {source_id} to {new_timeline_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to branch timeline in SQLite: {e}")
                return False

    async def detect_paradoxes(self, timeline_id: str) -> list[TemporalParadox]:
        """Detect paradoxes in SQLite timeline."""
        try:
            events = await self.get_events(timeline_id)
            paradoxes = []

            # Simple causality violation detection
            for i, event in enumerate(events):
                for _j, other_event in enumerate(events[i + 1 :], i + 1):
                    if self._is_causality_violation(event, other_event):
                        paradox = TemporalParadox(
                            paradox_id=str(uuid.uuid4()),
                            timeline_id=timeline_id,
                            paradox_type=ParadoxType.CAUSALITY_VIOLATION,
                            severity=0.8,
                            description=f"Causality violation between events {event.event_id} and {other_event.event_id}",
                            affected_events=[event.event_id, other_event.event_id],
                            detected_at=datetime.now(UTC),
                        )
                        paradoxes.append(paradox)
                        await self._store_paradox(paradox)

            return paradoxes

        except Exception as e:
            logger.error(f"Failed to detect paradoxes in SQLite: {e}")
            return []

    async def _store_paradox(self, paradox: TemporalParadox):
        """Store paradox in SQLite."""
        if not self._ensure_connection():

            return False

        cursor = self._connection.cursor()
        cursor.execute(
            """
            INSERT INTO temporal_paradoxes (
                paradox_id, timeline_id, paradox_type, severity, description,
                affected_events, resolved, resolution_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                paradox.paradox_id,
                paradox.timeline_id,
                paradox.paradox_type.value,
                paradox.severity,
                paradox.description,
                json.dumps(paradox.affected_events),
                paradox.resolved,
                paradox.resolution_method,
            ],
        )
        if self._connection is not None:

            self._connection.commit()

    def _calculate_event_checksum(self, event: TimelineEvent) -> str:
        """Calculate event checksum."""
        content = (
            f"{event.event_id}{event.timeline_id}{event.payload}{event.valid_at_us}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def _verify_event_integrity(
        self, event: TimelineEvent, stored_checksum: str
    ) -> bool:
        """Verify event integrity."""
        calculated_checksum = self._calculate_event_checksum(event)
        return calculated_checksum == stored_checksum

    def _is_causality_violation(
        self, event1: TimelineEvent, event2: TimelineEvent
    ) -> bool:
        """Check for causality violations."""
        if (
            event1.event_id == event2.parent_id
            and event1.valid_at_us > event2.valid_at_us
        ):
            return True
        return bool(
            event2.event_id == event1.parent_id
            and event2.valid_at_us > event1.valid_at_us
        )

    async def close(self):
        """Close SQLite connection."""
        if self._connection:
            self._connection.close()


def create_repository(config: TimelineServiceConfig) -> BaseRepository:
    """Factory function to create appropriate repository based on configuration."""
    if config.database.database_type == DatabaseType.SQLITE:
        return SQLiteRepository(config)
    else:
        if not CASSANDRA_AVAILABLE:
            logger.warning("Cassandra driver not available, falling back to SQLite")
            return SQLiteRepository(config)
        return ScyllaRepository(config)
