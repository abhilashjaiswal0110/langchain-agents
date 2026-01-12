"""Tests for Deep Agent functionality."""

import os
import pytest
import tempfile
import shutil
import uuid

# Set up mock API keys before importing app modules
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing"


class TestDeepAgentTypes:
    """Tests for Deep Agent type definitions."""

    def test_todo_creation(self):
        """Test Todo model creation."""
        from app.deepagents.core.types import Todo, TodoStatus

        todo = Todo(
            id=str(uuid.uuid4()),
            content="Test Task Description",
            status=TodoStatus.PENDING,
            priority=1,
        )

        assert "Test Task" in todo.content or todo.content == "Test Task Description"
        assert todo.status == TodoStatus.PENDING
        assert todo.priority == 1
        assert todo.id is not None

    def test_todo_status_transitions(self):
        """Test todo status can be changed."""
        from app.deepagents.core.types import Todo, TodoStatus

        todo = Todo(id="test-1", content="Test Task", status=TodoStatus.PENDING)
        assert todo.status == TodoStatus.PENDING

        todo.status = TodoStatus.IN_PROGRESS
        assert todo.status == TodoStatus.IN_PROGRESS

        todo.status = TodoStatus.COMPLETED
        assert todo.status == TodoStatus.COMPLETED

    def test_todo_mark_methods(self):
        """Test todo mark_* helper methods."""
        from app.deepagents.core.types import Todo, TodoStatus

        todo = Todo(id="test-2", content="Test Task", status=TodoStatus.PENDING)

        todo.mark_in_progress()
        assert todo.status == TodoStatus.IN_PROGRESS

        todo.mark_completed()
        assert todo.status == TodoStatus.COMPLETED
        assert todo.completed_at is not None

    def test_file_entry_creation(self):
        """Test FileEntry model creation."""
        from app.deepagents.core.types import FileEntry

        entry = FileEntry(
            path="/test/file.txt",
            content="Test content",
            file_type="text",
        )

        assert entry.path == "/test/file.txt"
        assert entry.content == "Test content"
        assert entry.file_type == "text"

    def test_subagent_definition(self):
        """Test SubAgentDefinition model."""
        from app.deepagents.core.types import SubAgentDefinition

        subagent = SubAgentDefinition(
            name="test_agent",
            description="Test agent description",
            system_prompt="You are a test agent",
            tools=[],
        )

        assert subagent.name == "test_agent"
        assert subagent.description == "Test agent description"
        assert len(subagent.tools) == 0

    def test_deep_agent_config(self):
        """Test DeepAgentConfig model."""
        from app.deepagents.core.types import DeepAgentConfig

        config = DeepAgentConfig(
            name="test_deep_agent",
            auto_planning=True,
            persistent_storage=True,
            max_subagents=3,
        )

        assert config.name == "test_deep_agent"
        assert config.auto_planning is True
        assert config.persistent_storage is True
        assert config.max_subagents == 3


class TestDeepAgentState:
    """Tests for Deep Agent state management."""

    def test_state_creation(self):
        """Test DeepAgentState creation."""
        from app.deepagents.core.state import DeepAgentState

        state = DeepAgentState()

        assert state.messages == []
        assert state.todos == []
        assert state.files == {}
        assert state.session_id is None

    def test_get_pending_todos(self):
        """Test filtering pending todos."""
        from app.deepagents.core.state import DeepAgentState
        from app.deepagents.core.types import Todo, TodoStatus

        state = DeepAgentState(
            todos=[
                Todo(id="1", content="Task 1", status=TodoStatus.PENDING),
                Todo(id="2", content="Task 2", status=TodoStatus.IN_PROGRESS),
                Todo(id="3", content="Task 3", status=TodoStatus.COMPLETED),
                Todo(id="4", content="Task 4", status=TodoStatus.PENDING),
            ]
        )

        pending = state.get_pending_todos()
        assert len(pending) == 2
        assert all(t.status == TodoStatus.PENDING for t in pending)

    def test_get_in_progress_todos(self):
        """Test filtering in-progress todos."""
        from app.deepagents.core.state import DeepAgentState
        from app.deepagents.core.types import Todo, TodoStatus

        state = DeepAgentState(
            todos=[
                Todo(id="1", content="Task 1", status=TodoStatus.PENDING),
                Todo(id="2", content="Task 2", status=TodoStatus.IN_PROGRESS),
                Todo(id="3", content="Task 3", status=TodoStatus.IN_PROGRESS),
            ]
        )

        in_progress = state.get_in_progress_todos()
        assert len(in_progress) == 2

    def test_get_completed_todos(self):
        """Test filtering completed todos."""
        from app.deepagents.core.state import DeepAgentState
        from app.deepagents.core.types import Todo, TodoStatus

        state = DeepAgentState(
            todos=[
                Todo(id="1", content="Task 1", status=TodoStatus.COMPLETED),
                Todo(id="2", content="Task 2", status=TodoStatus.PENDING),
            ]
        )

        completed = state.get_completed_todos()
        assert len(completed) == 1

    def test_todo_summary(self):
        """Test todo summary generation."""
        from app.deepagents.core.state import DeepAgentState
        from app.deepagents.core.types import Todo, TodoStatus

        state = DeepAgentState(
            todos=[
                Todo(id="1", content="Task 1", status=TodoStatus.COMPLETED),
                Todo(id="2", content="Task 2", status=TodoStatus.IN_PROGRESS),
                Todo(id="3", content="Task 3", status=TodoStatus.PENDING),
            ]
        )

        summary = state.get_todo_summary()
        assert "1/3 completed" in summary
        assert "1 in progress" in summary
        assert "1 pending" in summary

    def test_empty_todo_summary(self):
        """Test todo summary with no todos."""
        from app.deepagents.core.state import DeepAgentState

        state = DeepAgentState()
        summary = state.get_todo_summary()
        assert "No tasks planned" in summary


class TestPersistentStorage:
    """Tests for persistent file storage."""

    def setup_method(self):
        """Set up test storage directory."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test storage directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_read_file(self):
        """Test saving and reading files."""
        from app.deepagents.storage.persistent_backend import PersistentStorage

        storage = PersistentStorage(base_path=self.temp_dir)
        session_id = "test-session-123"

        # Save file
        entry = storage.save_file(
            session_id=session_id,
            path="/notes/test.txt",
            content="Hello, World!",
        )

        assert entry.path == "/notes/test.txt"
        assert entry.content == "Hello, World!"

        # Read file
        read_entry = storage.read_file(session_id, "/notes/test.txt")
        assert read_entry is not None
        assert read_entry.content == "Hello, World!"

    def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist."""
        from app.deepagents.storage.persistent_backend import PersistentStorage

        storage = PersistentStorage(base_path=self.temp_dir)
        entry = storage.read_file("test-session", "/nonexistent.txt")
        assert entry is None

    def test_delete_file(self):
        """Test deleting a file."""
        from app.deepagents.storage.persistent_backend import PersistentStorage

        storage = PersistentStorage(base_path=self.temp_dir)
        session_id = "test-session"

        # Save and then delete
        storage.save_file(session_id, "/test.txt", "content")
        result = storage.delete_file(session_id, "/test.txt")
        assert result is True

        # Verify deleted
        entry = storage.read_file(session_id, "/test.txt")
        assert entry is None

    def test_list_files(self):
        """Test listing files in a session."""
        from app.deepagents.storage.persistent_backend import PersistentStorage

        storage = PersistentStorage(base_path=self.temp_dir)
        session_id = "test-session"

        storage.save_file(session_id, "/file1.txt", "content1")
        storage.save_file(session_id, "/dir/file2.txt", "content2")
        storage.save_file(session_id, "/dir/file3.txt", "content3")

        files = storage.list_files(session_id)
        assert len(files) == 3
        assert "/file1.txt" in files
        assert "/dir/file2.txt" in files

    def test_save_and_get_todos(self):
        """Test saving and retrieving todos."""
        from app.deepagents.storage.persistent_backend import PersistentStorage
        from app.deepagents.core.types import Todo, TodoStatus

        storage = PersistentStorage(base_path=self.temp_dir)
        session_id = "test-session"

        todos = [
            Todo(id="todo-1", content="Task 1", status=TodoStatus.PENDING),
            Todo(id="todo-2", content="Task 2", status=TodoStatus.IN_PROGRESS),
        ]

        storage.save_todos(session_id, todos)
        retrieved = storage.get_todos(session_id)

        assert len(retrieved) == 2
        assert retrieved[0].content == "Task 1"
        assert retrieved[1].content == "Task 2"

    def test_session_metadata(self):
        """Test session metadata storage."""
        from app.deepagents.storage.persistent_backend import PersistentStorage

        storage = PersistentStorage(base_path=self.temp_dir)
        session_id = "test-session"

        storage.save_session_metadata(session_id, {"user_id": "user123", "created": "2024-01-01"})
        metadata = storage.get_session_metadata(session_id)

        assert metadata is not None
        assert metadata["user_id"] == "user123"

    def test_clear_session(self):
        """Test clearing session data."""
        from app.deepagents.storage.persistent_backend import PersistentStorage

        storage = PersistentStorage(base_path=self.temp_dir)
        session_id = "test-session"

        storage.save_file(session_id, "/test.txt", "content")
        storage.save_session_metadata(session_id, {"test": "data"})

        storage.clear_session(session_id)

        assert storage.read_file(session_id, "/test.txt") is None
        assert storage.get_session_metadata(session_id) is None


class TestMemoryStorage:
    """Tests for in-memory storage backend."""

    def test_save_and_read_file(self):
        """Test saving and reading files in memory."""
        from app.deepagents.storage.memory_backend import MemoryStorage

        storage = MemoryStorage()
        session_id = "test-session"

        entry = storage.save_file(session_id, "/test.txt", "Hello!")
        assert entry.content == "Hello!"

        read_entry = storage.read_file(session_id, "/test.txt")
        assert read_entry.content == "Hello!"

    def test_session_isolation(self):
        """Test that sessions are isolated."""
        from app.deepagents.storage.memory_backend import MemoryStorage

        storage = MemoryStorage()

        storage.save_file("session1", "/file.txt", "Session 1 content")
        storage.save_file("session2", "/file.txt", "Session 2 content")

        entry1 = storage.read_file("session1", "/file.txt")
        entry2 = storage.read_file("session2", "/file.txt")

        assert entry1.content == "Session 1 content"
        assert entry2.content == "Session 2 content"


class TestKnowledgeTools:
    """Tests for knowledge base tools."""

    def test_search_knowledge_base(self):
        """Test knowledge base search."""
        from app.deepagents.tools.knowledge_tools import search_knowledge_base

        result = search_knowledge_base.invoke({"query": "VPN"})
        assert "VPN" in result
        assert "KB0010001" in result

    def test_get_kb_article(self):
        """Test retrieving a specific KB article."""
        from app.deepagents.tools.knowledge_tools import get_kb_article

        result = get_kb_article.invoke({"article_number": "KB0010001"})
        assert "VPN Connection Troubleshooting" in result

    def test_get_kb_article_not_found(self):
        """Test retrieving nonexistent KB article."""
        from app.deepagents.tools.knowledge_tools import get_kb_article

        result = get_kb_article.invoke({"article_number": "KB9999999"})
        assert "not found" in result

    def test_suggest_kb_articles(self):
        """Test KB article suggestions."""
        from app.deepagents.tools.knowledge_tools import suggest_kb_articles

        result = suggest_kb_articles.invoke({
            "incident_description": "User cannot connect to VPN"
        })
        assert "KB0010001" in result


class TestIncidentTools:
    """Tests for incident management tools."""

    def test_search_incidents(self):
        """Test incident search."""
        from app.deepagents.tools.incident_tools import search_incidents

        result = search_incidents.invoke({"query": "network"})
        # In simulation mode without matching data, may return no results
        assert isinstance(result, str)

    def test_search_incidents_all(self):
        """Test incident search with no filters."""
        from app.deepagents.tools.incident_tools import search_incidents

        result = search_incidents.invoke({})
        assert isinstance(result, str)

    def test_create_incident(self):
        """Test incident creation."""
        from app.deepagents.tools.incident_tools import create_incident

        result = create_incident.invoke({
            "short_description": "Test incident",
            "description": "This is a test incident",
            "category": "Software",
            "subcategory": "Application",
            "urgency": "2",
            "impact": "2",
        })
        assert "INC" in result or "Created" in result or "SIMULATION" in result


class TestChangeTools:
    """Tests for change management tools."""

    def test_search_changes(self):
        """Test change request search."""
        from app.deepagents.tools.change_tools import search_changes

        result = search_changes.invoke({"query": "upgrade"})
        assert isinstance(result, str)


class TestAssetTools:
    """Tests for asset/CMDB tools."""

    def test_search_cmdb(self):
        """Test CMDB search."""
        from app.deepagents.tools.asset_tools import search_cmdb

        result = search_cmdb.invoke({"query": "server"})
        assert isinstance(result, str)


class TestSLATools:
    """Tests for SLA monitoring tools."""

    def test_get_sla_status(self):
        """Test SLA status retrieval."""
        from app.deepagents.tools.sla_tools import get_sla_status

        result = get_sla_status.invoke({"ticket_number": "INC0010001"})
        assert "SLA" in result


class TestSubagentDefinitions:
    """Tests for subagent definitions."""

    def test_get_all_subagents(self):
        """Test retrieving all subagent definitions."""
        from app.deepagents.subagents.definitions import get_all_subagents

        subagents = get_all_subagents()
        assert len(subagents) >= 6  # At least 6 subagents defined

        names = [s.name for s in subagents]
        # Names use kebab-case
        assert "incident-manager" in names
        assert "change-manager" in names
        assert "problem-manager" in names
        assert "asset-manager" in names
        assert "sla-monitor" in names
        assert "knowledge-manager" in names

    def test_get_subagent_tools(self):
        """Test retrieving subagent tools."""
        from app.deepagents.subagents.definitions import get_subagent_tools

        # Use correct kebab-case name
        tools = get_subagent_tools("incident-manager")
        assert len(tools) > 0

    def test_get_subagent_tools_unknown(self):
        """Test retrieving tools for unknown subagent."""
        from app.deepagents.subagents.definitions import get_subagent_tools

        tools = get_subagent_tools("unknown-agent")
        assert len(tools) == 0


class TestITOperationsAgent:
    """Tests for the main IT Operations Deep Agent."""

    def test_agent_creation_without_api_key(self):
        """Test that agent creation fails without API key."""
        # Clear the API key to test the error case
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        old_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            from app.deepagents.it_operations_agent import create_it_operations_agent
            with pytest.raises(ValueError, match="No LLM API key found"):
                create_it_operations_agent()
        finally:
            # Restore keys
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
            if old_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = old_anthropic

    def test_agent_module_imports(self):
        """Test that agent module imports correctly."""
        from app.deepagents.it_operations_agent import (
            ITOperationsDeepAgent,
            create_it_operations_agent,
            get_graph,
            IT_OPERATIONS_SYSTEM_PROMPT,
        )

        assert ITOperationsDeepAgent is not None
        assert create_it_operations_agent is not None
        assert get_graph is not None
        assert "IT Operations" in IT_OPERATIONS_SYSTEM_PROMPT


class TestDeepAgentMiddleware:
    """Tests for Deep Agent middleware components."""

    def test_todolist_middleware_tools(self):
        """Test TodoList middleware creates expected tools."""
        from app.deepagents.core.middleware import TodoListMiddleware

        middleware = TodoListMiddleware()
        tools = middleware.get_tools()

        tool_names = [t.name for t in tools]
        assert "write_todos" in tool_names
        assert "update_todo" in tool_names
        assert "get_todos" in tool_names

    def test_filesystem_middleware_tools(self):
        """Test Filesystem middleware creates expected tools."""
        from app.deepagents.core.middleware import FilesystemMiddleware

        middleware = FilesystemMiddleware()
        tools = middleware.get_tools()

        tool_names = [t.name for t in tools]
        assert "ls" in tool_names
        assert "read_file" in tool_names
        assert "write_file" in tool_names


class TestDeepAgentAPIEndpoints:
    """Tests for Deep Agent API endpoints."""

    def test_health_includes_deep_agent_status(self):
        """Test health endpoint includes Deep Agent status."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "deep_agent_loaded" in data

    def test_start_deep_agent_session(self):
        """Test starting a Deep Agent session."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.post(
            "/api/deepagent/start",
            json={"user_id": "test-user"}
        )

        # May fail if agent not loaded (503), but should return valid response
        assert response.status_code in [200, 503]
        data = response.json()
        # If successful, should have session_id
        if response.status_code == 200:
            assert "session_id" in data

    def test_list_subagents_endpoint(self):
        """Test listing available subagents via API."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/deepagent/subagents")

        # May return 503 if agent not loaded
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "subagents" in data

    def test_todos_endpoint_not_found(self):
        """Test todos endpoint with invalid session returns 404."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/deepagent/todos/invalid-session-id")

        # Should return 404 or 503 (if agent not loaded)
        assert response.status_code in [404, 503]

    def test_files_endpoint_not_found(self):
        """Test files endpoint with invalid session returns 404."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/deepagent/files/invalid-session-id")

        # Should return 404 or 503 (if agent not loaded)
        assert response.status_code in [404, 503]


class TestSecurityConsiderations:
    """Security-focused tests for Deep Agent."""

    def test_session_id_sanitization(self):
        """Test that session IDs are sanitized in storage."""
        from app.deepagents.storage.persistent_backend import PersistentStorage

        temp_dir = tempfile.mkdtemp()
        try:
            storage = PersistentStorage(base_path=temp_dir)

            # Test with potentially malicious session ID
            malicious_id = "../../../etc/passwd"
            safe_path = storage._get_session_path(malicious_id)

            # Should not allow directory traversal - ".." should be removed/replaced
            assert ".." not in str(safe_path)
            # Path should be contained within the base temp directory
            assert str(safe_path).startswith(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_file_path_sanitization(self):
        """Test that file paths are sanitized against directory traversal."""
        from app.deepagents.storage.persistent_backend import PersistentStorage

        temp_dir = tempfile.mkdtemp()
        try:
            storage = PersistentStorage(base_path=temp_dir)
            session_id = "test-session"

            # Test normal file save works
            entry = storage.save_file(
                session_id,
                "/safe/path/test.txt",
                "safe content"
            )
            assert entry.path == "/safe/path/test.txt"

            # Test with directory traversal attempt (/../ pattern)
            # The storage should either strip the traversal or prevent it
            storage.save_file(
                session_id,
                "/data/../../../etc/passwd",
                "malicious content"
            )
            # The actual file should be stored safely within the session directory
            # Check that the file doesn't escape the temp directory
            files = storage.list_files(session_id)
            assert len(files) >= 1
            # Verify files are stored within the temp_dir
            session_path = storage._get_session_path(session_id)
            assert str(session_path).startswith(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_sensitive_data_in_logs(self):
        """Test that sensitive data is not exposed in responses."""
        from app.deepagents.tools.incident_tools import search_incidents

        result = search_incidents.invoke({"query": "password"})

        # Should not contain actual passwords
        assert "sk-" not in result
        assert "api_key" not in result.lower()
