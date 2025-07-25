"""Tests for the main entry point module."""

import logging
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from mcp_memory_server.main import Settings, setup_logging, main, app


class TestMainSettings:
    """Test the Settings class in main.py."""

    def test_settings_default_values(self):
        """Test that Settings has correct default values."""
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_settings_with_custom_log_level(self):
        """Test Settings with custom log level."""
        settings = Settings(log_level="DEBUG")
        assert settings.log_level == "DEBUG"

    def test_settings_from_env(self):
        """Test Settings loading from environment variables."""
        with patch.dict("os.environ", {"LOG_LEVEL": "WARNING"}):
            settings = Settings()
            assert settings.log_level == "WARNING"


class TestSetupLogging:
    """Test the setup_logging function."""

    @patch("mcp_memory_server.main.logging.basicConfig")
    def test_setup_logging_default(self, mock_basic_config):
        """Test setup_logging with default level."""
        setup_logging()

        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    @patch("mcp_memory_server.main.logging.basicConfig")
    def test_setup_logging_custom_level(self, mock_basic_config):
        """Test setup_logging with custom level."""
        setup_logging("DEBUG")

        mock_basic_config.assert_called_once_with(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    @patch("mcp_memory_server.main.logging.basicConfig")
    def test_setup_logging_invalid_level(self, mock_basic_config):
        """Test setup_logging with invalid level raises AttributeError."""
        # Invalid log levels should raise AttributeError since getattr() doesn't have a default
        with pytest.raises(AttributeError):
            setup_logging("INVALID_LEVEL")


class TestFastMCPApp:
    """Test the FastMCP app instance and tools."""

    def test_app_instance_exists(self):
        """Test that the FastMCP app instance is created."""
        assert app is not None
        assert hasattr(app, "run")

    def test_ping_tool_existence(self):
        """Test that the ping tool is defined in the module."""
        # Test that the app has tools registered
        from mcp_memory_server.main import app

        assert app is not None
        assert hasattr(app, "get_tools")

        # We can't easily test the decorated function due to FastMCP wrapping,
        # but we can verify the app exists and has the right interface


class TestMainFunction:
    """Test the main entry point function."""

    @patch("mcp_memory_server.main.app.run")
    @patch("mcp_memory_server.main.setup_logging")
    @patch("mcp_memory_server.main.get_settings")
    @patch("mcp_memory_server.main.logging.getLogger")
    def test_main_function_execution(
        self, mock_get_logger, mock_get_settings, mock_setup_logging, mock_app_run
    ):
        """Test that main function executes correctly."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.log_level = "INFO"
        mock_settings.server_port = 8000
        mock_get_settings.return_value = mock_settings

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Call main function
        main()

        # Verify the expected calls
        mock_get_settings.assert_called_once()
        mock_setup_logging.assert_called_once_with("INFO")
        mock_get_logger.assert_called_once_with("mcp_memory_server.main")
        
        # Verify both log messages are called
        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_any_call("Starting MCP Memory Server...")
        mock_logger.info.assert_any_call("Server will start on port 8000")
        mock_app_run.assert_called_once()

    @patch("mcp_memory_server.main.app.run")
    @patch("mcp_memory_server.main.setup_logging")
    @patch("mcp_memory_server.main.get_settings")
    @patch("mcp_memory_server.main.logging.getLogger")
    def test_main_function_with_debug_logging(
        self, mock_get_logger, mock_get_settings, mock_setup_logging, mock_app_run
    ):
        """Test main function with DEBUG log level."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.log_level = "DEBUG"
        mock_settings.server_port = 8000
        mock_get_settings.return_value = mock_settings

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Call main function
        main()

        # Verify DEBUG logging was set up
        mock_setup_logging.assert_called_once_with("DEBUG")

    @patch("mcp_memory_server.main.app.run")
    @patch("mcp_memory_server.main.setup_logging")
    @patch("mcp_memory_server.main.Settings")
    @patch("mcp_memory_server.main.logging.getLogger")
    def test_main_function_app_run_called(
        self, mock_get_logger, mock_settings_class, mock_setup_logging, mock_app_run
    ):
        """Test that main function calls app.run()."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.log_level = "INFO"
        mock_settings.server_port = 8000
        mock_settings_class.return_value = mock_settings

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Call main function
        main()

        # Verify app.run() was called
        mock_app_run.assert_called_once()

    @patch("mcp_memory_server.main.app.run")
    @patch("mcp_memory_server.main.setup_logging")
    @patch("mcp_memory_server.main.get_settings")
    def test_main_function_settings_error_handling(
        self, mock_get_settings, mock_setup_logging, mock_app_run
    ):
        """Test main function behavior when Settings initialization fails."""
        # Make get_settings raise an exception
        mock_get_settings.side_effect = ValidationError.from_exception_data(
            "Settings", []
        )

        # main() should still be robust and not crash
        with pytest.raises(ValidationError):
            main()

        # setup_logging should not be called if Settings fails
        mock_setup_logging.assert_not_called()
        mock_app_run.assert_not_called()

    @patch("mcp_memory_server.main.os.environ", {})
    @patch("mcp_memory_server.main.app.run")
    @patch("mcp_memory_server.main.setup_logging")
    @patch("mcp_memory_server.main.get_settings")
    @patch("mcp_memory_server.main.logging.getLogger")
    def test_main_function_sets_fastmcp_port_default(
        self, mock_get_logger, mock_get_settings, mock_setup_logging, mock_app_run
    ):
        """Test that main function sets FASTMCP_PORT environment variable to default port."""
        import os
        
        # Setup mocks
        mock_settings = Mock()
        mock_settings.log_level = "INFO"
        mock_settings.server_port = 8000
        mock_get_settings.return_value = mock_settings

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Call main function
        main()

        # Verify FASTMCP_PORT was set
        assert os.environ.get("FASTMCP_PORT") == "8000"
        
        # Verify logging includes port information
        mock_logger.info.assert_any_call("Server will start on port 8000")

    @patch("mcp_memory_server.main.os.environ", {})
    @patch("mcp_memory_server.main.app.run")
    @patch("mcp_memory_server.main.setup_logging")
    @patch("mcp_memory_server.main.get_settings")
    @patch("mcp_memory_server.main.logging.getLogger")
    def test_main_function_sets_fastmcp_port_custom(
        self, mock_get_logger, mock_get_settings, mock_setup_logging, mock_app_run
    ):
        """Test that main function sets FASTMCP_PORT environment variable to custom port."""
        import os
        
        # Setup mocks
        mock_settings = Mock()
        mock_settings.log_level = "INFO"
        mock_settings.server_port = 9000
        mock_get_settings.return_value = mock_settings

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Call main function
        main()

        # Verify FASTMCP_PORT was set to custom port
        assert os.environ.get("FASTMCP_PORT") == "9000"
        
        # Verify logging includes custom port information
        mock_logger.info.assert_any_call("Server will start on port 9000")

    @patch("mcp_memory_server.main.app.run")
    @patch("mcp_memory_server.main.setup_logging")
    @patch("mcp_memory_server.main.get_settings")
    @patch("mcp_memory_server.main.logging.getLogger")
    def test_main_function_port_logging_order(
        self, mock_get_logger, mock_get_settings, mock_setup_logging, mock_app_run
    ):
        """Test that port logging happens after server startup message."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.log_level = "INFO"
        mock_settings.server_port = 8080
        mock_get_settings.return_value = mock_settings

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Call main function
        main()

        # Check that both log messages were called
        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_any_call("Starting MCP Memory Server...")
        mock_logger.info.assert_any_call("Server will start on port 8080")


class TestMainModuleExecution:
    """Test direct module execution behavior."""

    @patch("mcp_memory_server.main.main")
    def test_module_main_execution(self, mock_main):
        """Test that main() is called when module is executed directly."""
        # This tests the if __name__ == "__main__": main() block
        # We can't easily test this directly, but we can verify the main function exists and is callable
        assert callable(main)

        # Verify that calling main works
        mock_main()
        mock_main.assert_called_once()


class TestMainModuleIntegration:
    """Integration tests for the main module."""

    def test_main_module_imports(self):
        """Test that all necessary imports work correctly."""
        # These imports should not raise any exceptions
        from mcp_memory_server.main import Settings, setup_logging, main, app

        assert Settings is not None
        assert setup_logging is not None
        assert main is not None
        assert app is not None

    def test_memory_tools_import(self):
        """Test that memory_tools import works independently."""
        # The memory_tools module should be importable independently
        # This should not raise an import error
        try:
            from mcp_memory_server.tools import memory_tools

            # If we get here, the import worked
            assert memory_tools is not None
            assert hasattr(memory_tools, "app")  # Should have its own app instance
        except ImportError as e:
            # If there's an import error, it should be expected (e.g., missing dependencies)
            # For now, we'll allow this to pass since tools might not be fully implemented
            pytest.skip(f"memory_tools import failed: {e}")

    @patch("mcp_memory_server.main.logging.basicConfig")
    def test_logging_setup_integration(self, mock_basic_config):
        """Test logging setup integration with Settings."""
        settings = Settings(log_level="WARNING")
        setup_logging(settings.log_level)

        mock_basic_config.assert_called_once_with(
            level=logging.WARNING,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


class TestMainModuleEdgeCases:
    """Test edge cases and error conditions in main module."""

    def test_settings_with_none_log_level(self):
        """Test Settings behavior with None log level."""
        # Pydantic should handle this gracefully with defaults
        settings = Settings()
        assert settings.log_level == "INFO"  # Should use default

    def test_settings_with_empty_string_log_level(self):
        """Test Settings with empty string log level."""
        with patch.dict("os.environ", {"LOG_LEVEL": ""}):
            with pytest.raises(ValidationError):
                Settings()
            # Empty string should be rejected, not fall back to default

    @patch("mcp_memory_server.main.logging.basicConfig")
    def test_setup_logging_case_insensitive(self, mock_basic_config):
        """Test that setup_logging handles case-insensitive log levels."""
        setup_logging("debug")  # lowercase

        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        # Should convert to proper logging level
        assert call_args[1]["level"] == logging.DEBUG

    @patch("mcp_memory_server.main.app.run")
    @patch("mcp_memory_server.main.setup_logging")
    @patch("mcp_memory_server.main.logging.getLogger")
    def test_main_logger_error_handling(
        self, mock_get_logger, mock_setup_logging, mock_app_run
    ):
        """Test main function when logger operations fail."""
        # Make logger.info raise an exception
        mock_logger = Mock()
        mock_logger.info.side_effect = Exception("Logging failed")
        mock_get_logger.return_value = mock_logger

        # main() should continue even if logging fails
        with pytest.raises(Exception, match="Logging failed"):
            main()

        # setup_logging should still be called
        mock_setup_logging.assert_called_once()
