"""Tests for the Settings configuration class."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from mcp_memory_server.config.settings import Settings, get_settings, reload_settings


class TestSettingsDefaults:
    """Test Settings class default values."""

    def test_default_values(self):
        """Test that default settings values are correct."""
        # Ensure we test defaults by temporarily removing SERVER_PORT env var if it exists
        with patch.dict(os.environ, {}, clear=False):
            # Remove SERVER_PORT to test actual defaults
            if "SERVER_PORT" in os.environ:
                del os.environ["SERVER_PORT"]
            
            settings = Settings()

            assert settings.storage_backend == "chroma"
            # Path should be resolved to absolute path
            assert "data" in settings.chroma_path
            assert "chroma_db" in settings.chroma_path
            assert settings.chroma_collection_name == "memories"
            assert settings.embedding_provider == "ollama"
            assert settings.ollama_base_url == "http://localhost:11434"
            assert settings.ollama_model == "mxbai-embed-large"
            assert settings.max_memories_per_request == 100
            assert settings.default_search_limit == 10
            assert settings.similarity_threshold == 0.7
            assert settings.log_level == "INFO"
            assert settings.server_port == 8139

    def test_settings_with_env_variables(self):
        """Test settings loading from environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = str(Path(temp_dir) / "custom_chroma")

            env_vars = {
                "STORAGE_BACKEND": "chroma",
                "CHROMA_PATH": custom_path,
                "CHROMA_COLLECTION_NAME": "custom_memories",
                "EMBEDDING_PROVIDER": "ollama",
                "OLLAMA_BASE_URL": "http://custom:11434",
                "OLLAMA_MODEL": "custom-model",
                "MAX_MEMORIES_PER_REQUEST": "200",
                "DEFAULT_SEARCH_LIMIT": "20",
                "SIMILARITY_THRESHOLD": "0.8",
                "LOG_LEVEL": "DEBUG",
                "SERVER_PORT": "9000",
            }

            with patch.dict(os.environ, env_vars):
                settings = Settings()

                assert settings.storage_backend == "chroma"
                assert "custom_chroma" in settings.chroma_path  # Resolved path
                assert settings.chroma_collection_name == "custom_memories"
                assert settings.embedding_provider == "ollama"
                assert settings.ollama_base_url == "http://custom:11434"
                assert settings.ollama_model == "custom-model"
                assert settings.max_memories_per_request == 200
                assert settings.default_search_limit == 20
                assert settings.similarity_threshold == 0.8
                assert settings.log_level == "DEBUG"
                assert settings.server_port == 9000


class TestSettingsValidation:
    """Test Settings field validation."""

    def test_storage_backend_validation(self):
        """Test storage backend validation (MVP: only chroma allowed)."""
        # Valid value
        settings = Settings(storage_backend="chroma")
        assert settings.storage_backend == "chroma"

        # Invalid value should raise validation error
        with pytest.raises(ValidationError):
            Settings(storage_backend="invalid")

    def test_embedding_provider_validation(self):
        """Test embedding provider validation (MVP: only ollama allowed)."""
        # Valid value
        settings = Settings(embedding_provider="ollama")
        assert settings.embedding_provider == "ollama"

        # Invalid value should raise validation error
        with pytest.raises(ValidationError):
            Settings(embedding_provider="invalid")

    def test_max_memories_per_request_validation(self):
        """Test max_memories_per_request field validation."""
        # Valid values
        settings = Settings(max_memories_per_request=1)
        assert settings.max_memories_per_request == 1

        settings = Settings(max_memories_per_request=1000)
        assert settings.max_memories_per_request == 1000

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(max_memories_per_request=0)  # Too low

        with pytest.raises(ValidationError):
            Settings(max_memories_per_request=1001)  # Too high

    def test_default_search_limit_validation(self):
        """Test default_search_limit field validation."""
        # Valid values
        settings = Settings(default_search_limit=1)
        assert settings.default_search_limit == 1

        settings = Settings(default_search_limit=100)
        assert settings.default_search_limit == 100

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(default_search_limit=0)  # Too low

        with pytest.raises(ValidationError):
            Settings(default_search_limit=101)  # Too high

    def test_similarity_threshold_validation(self):
        """Test similarity_threshold field validation."""
        # Valid values
        settings = Settings(similarity_threshold=0.0)
        assert settings.similarity_threshold == 0.0

        settings = Settings(similarity_threshold=1.0)
        assert settings.similarity_threshold == 1.0

        settings = Settings(similarity_threshold=0.5)
        assert settings.similarity_threshold == 0.5

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(similarity_threshold=-0.1)  # Too low

        with pytest.raises(ValidationError):
            Settings(similarity_threshold=1.1)  # Too high

    def test_server_port_validation(self):
        """Test server_port field validation."""
        # Valid values
        settings = Settings(server_port=1024)  # Minimum port
        assert settings.server_port == 1024

        settings = Settings(server_port=65535)  # Maximum port
        assert settings.server_port == 65535

        settings = Settings(server_port=8139)  # Default port
        assert settings.server_port == 8139

        settings = Settings(server_port=9000)  # Custom port
        assert settings.server_port == 9000

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(server_port=1023)  # Too low (below 1024)

        with pytest.raises(ValidationError):
            Settings(server_port=65536)  # Too high (above 65535)

        with pytest.raises(ValidationError):
            Settings(server_port=0)  # Too low

        with pytest.raises(ValidationError):
            Settings(server_port=-1)  # Negative


class TestSettingsFieldValidators:
    """Test Settings custom field validators."""

    def test_chroma_path_validator_creates_directory(self):
        """Test that chroma_path validator creates directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test_chroma"

            # Directory doesn't exist yet
            assert not test_path.exists()

            # Settings should create it
            settings = Settings(chroma_path=str(test_path))

            # Directory should now exist
            assert Path(settings.chroma_path).exists()
            assert Path(settings.chroma_path).is_dir()

    def test_chroma_path_validator_invalid_path(self):
        """Test chroma_path validator with invalid path."""
        # Try to create directory in a location that doesn't exist and can't be created
        # Use a path with invalid characters for Windows
        invalid_path = "C:\\\\invalid<>path|"

        with pytest.raises(ValidationError) as exc_info:
            Settings(chroma_path=invalid_path)

        error_messages = str(exc_info.value)
        assert "Cannot create ChromaDB directory" in error_messages

    def test_ollama_base_url_validator_valid_urls(self):
        """Test ollama_base_url validator with valid URLs."""
        # Test valid URLs
        valid_urls = [
            "http://localhost:11434",
            "https://localhost:11434",
            "http://127.0.0.1:11434",
            "https://remote-server.com:11434",
            "http://localhost:11434/",  # Should remove trailing slash
        ]

        for url in valid_urls:
            settings = Settings(ollama_base_url=url)
            # Should remove trailing slash
            assert not settings.ollama_base_url.endswith("/")
            assert settings.ollama_base_url.startswith(("http://", "https://"))

    def test_ollama_base_url_validator_invalid_urls(self):
        """Test ollama_base_url validator with invalid URLs."""
        invalid_urls = [
            "localhost:11434",  # Missing protocol
            "ftp://localhost:11434",  # Wrong protocol
            "not-a-url",
            "",
        ]

        for url in invalid_urls:
            with pytest.raises(ValidationError) as exc_info:
                Settings(ollama_base_url=url)

            error_messages = str(exc_info.value)
            assert "must start with http:// or https://" in error_messages

    def test_chroma_collection_name_validator_valid_names(self):
        """Test chroma_collection_name validator with valid names."""
        valid_names = [
            "memories",
            "test_collection",
            "test-collection",
            "collection123",
            "my_test_collection_2025",
        ]

        for name in valid_names:
            settings = Settings(chroma_collection_name=name)
            assert settings.chroma_collection_name == name

    def test_chroma_collection_name_validator_invalid_names(self):
        """Test chroma_collection_name validator with invalid names."""
        invalid_names = [
            "",  # Empty
            "collection with spaces",  # Spaces
            "collection@special",  # Special characters
            "collection.with.dots",  # Dots
            "collection/with/slashes",  # Slashes
        ]

        for name in invalid_names:
            with pytest.raises(ValidationError) as exc_info:
                Settings(chroma_collection_name=name)

            error_messages = str(exc_info.value)
            assert "alphanumeric characters, hyphens, and underscores" in error_messages

    def test_log_level_validator_valid_levels(self):
        """Test log_level validator with valid levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            settings = Settings(log_level=level)
            assert settings.log_level == level

        # Test case insensitive
        for level in ["debug", "info", "warning", "error", "critical"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level.upper()

    def test_log_level_validator_invalid_levels(self):
        """Test log_level validator with invalid levels."""
        invalid_levels = ["TRACE", "VERBOSE", "invalid", ""]

        for level in invalid_levels:
            with pytest.raises(ValidationError) as exc_info:
                Settings(log_level=level)

            error_messages = str(exc_info.value)
            assert "Log level must be one of:" in error_messages


class TestSettingsMethods:
    """Test Settings class methods."""

    def test_get_chroma_path_returns_path_object(self):
        """Test get_chroma_path returns Path object."""
        settings = Settings()
        chroma_path = settings.get_chroma_path()

        assert isinstance(chroma_path, Path)
        assert str(chroma_path) == settings.chroma_path

    def test_get_ollama_embed_url(self):
        """Test get_ollama_embed_url constructs correct URL."""
        settings = Settings(ollama_base_url="http://localhost:11434")
        embed_url = settings.get_ollama_embed_url()

        assert embed_url == "http://localhost:11434/api/embeddings"

        # Test with different base URL
        settings = Settings(ollama_base_url="https://remote:8080")
        embed_url = settings.get_ollama_embed_url()

        assert embed_url == "https://remote:8080/api/embeddings"

    def test_str_representation(self):
        """Test string representation of Settings."""
        settings = Settings()
        str_repr = str(settings)

        assert "Settings(" in str_repr
        assert "storage_backend=chroma" in str_repr
        assert "embedding_provider=ollama" in str_repr
        assert "ollama_model=mxbai-embed-large" in str_repr
        assert "chroma_path=" in str_repr


class TestSettingsSingleton:
    """Test Settings singleton functions."""

    def test_get_settings_returns_same_instance(self):
        """Test that get_settings returns the same instance."""
        # First call
        settings1 = get_settings()

        # Second call should return same instance
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reload_settings_creates_new_instance(self):
        """Test that reload_settings creates a new instance."""
        # Get initial settings
        settings1 = get_settings()

        # Reload should create new instance
        settings2 = reload_settings()

        # Should be different objects but same values
        assert settings1 is not settings2
        assert settings1.storage_backend == settings2.storage_backend

        # get_settings should now return the new instance
        settings3 = get_settings()
        assert settings2 is settings3

    def test_singleton_with_environment_changes(self):
        """Test singleton behavior with environment variable changes."""
        # Set initial environment
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            settings1 = reload_settings()  # Force new instance
            assert settings1.log_level == "DEBUG"

        # Change environment
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            # get_settings should still return old instance
            settings2 = get_settings()
            assert settings2 is settings1
            assert settings2.log_level == "DEBUG"  # Still old value

            # reload_settings should pick up new environment
            settings3 = reload_settings()
            assert settings3 is not settings1
            assert settings3.log_level == "ERROR"  # New value


class TestSettingsEdgeCases:
    """Test Settings edge cases and error conditions."""

    def test_settings_with_empty_env_file(self):
        """Test Settings with empty .env file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("")  # Empty file
            f.flush()
            f.close()  # Close file handle on Windows

            try:
                # Should use defaults when .env is empty
                with patch.dict(os.environ, {"ENV_FILE": f.name}):
                    settings = Settings()
                    assert settings.storage_backend == "chroma"
                    assert settings.log_level == "INFO"
            finally:
                try:
                    os.unlink(f.name)
                except (FileNotFoundError, PermissionError):
                    pass  # Ignore if file is already deleted or locked

    def test_settings_with_partial_env_variables(self):
        """Test Settings with only some environment variables set."""
        env_vars = {"OLLAMA_MODEL": "custom-model", "LOG_LEVEL": "WARNING"}

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()

            # Changed values
            assert settings.ollama_model == "custom-model"
            assert settings.log_level == "WARNING"

            # Default values for others
            assert settings.storage_backend == "chroma"
            assert settings.max_memories_per_request == 100

    def test_settings_case_insensitive_env_vars(self):
        """Test that environment variables are case-insensitive."""
        env_vars = {
            "log_level": "debug",  # lowercase
            "OLLAMA_MODEL": "TEST-MODEL",  # uppercase
            "Max_Memories_Per_Request": "50",  # mixed case
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings()

            assert settings.log_level == "DEBUG"
            assert settings.ollama_model == "TEST-MODEL"
            assert settings.max_memories_per_request == 50

    def test_invalid_numeric_env_variables(self):
        """Test Settings with invalid numeric environment variables."""
        with patch.dict(os.environ, {"MAX_MEMORIES_PER_REQUEST": "not-a-number"}):
            with pytest.raises(ValidationError):
                Settings()

        with patch.dict(os.environ, {"SIMILARITY_THRESHOLD": "invalid"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_settings_with_special_characters_in_paths(self):
        """Test Settings with special characters in paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create path with spaces (should work)
            test_path = Path(temp_dir) / "path with spaces"

            settings = Settings(chroma_path=str(test_path))
            assert Path(settings.chroma_path).exists()
            assert "path with spaces" in settings.chroma_path
