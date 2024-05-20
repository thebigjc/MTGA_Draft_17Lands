import pytest
import logging
from src.overlay import start_overlay
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def catch_log_errors(caplog):
    """
    Verify that the app is not generating any logging errors. 
    
    This function catches any logging errors in the app by checking log records. 
    If any errors are found, it fails the test with a descriptive message.
    """
    yield
    errors = [record for record in caplog.get_records("call") if record.levelno >= logging.ERROR]
    assert not errors, f"Log error detected - fix any errors that appear in the captured log call"
            
@pytest.fixture(name="mock_scanner")
def fixture_mock_scanner():
    """
    Mock the ArenaScanner class and all of its methods within overlay.py.
    """
    mock_instance = MagicMock()
    mock_instance.retrieve_color_win_rate.return_value = {"Auto": 0.0}
    mock_instance.retrieve_data_sources.return_value = {"None" : ""}
    mock_instance.retrieve_tier_source.return_value = []
    mock_instance.retrieve_set_metrics.return_value = None
    mock_instance.retrieve_tier_data.return_value = ({},{})
    mock_instance.draft_start_search.return_value = False
    mock_instance.retrieve_current_pack_and_pick.return_value = (0,0)
    mock_instance.retrieve_current_limited_event.return_value = ("","")  
    yield mock_instance
    
def test_overlay_tkinter_pass(mock_scanner):
    """
    Verify that the app starts up without generating exceptions or logging errors.
    
    This function will catch all exceptions that occur during startup. 
    The mainloop function is mocked so that the overlay immediately exits after startup.
    AppUpdate and messagebox are mocked to prevent the app from opening a window and blocking the test.
    """
    with (
        patch("tkinter.Tk.mainloop", return_value=None),
        patch("tkinter.messagebox.showinfo", return_value=None),
        patch("src.overlay.stat", return_value=MagicMock(st_mtime=0)),
        patch("src.overlay.write_configuration", return_value=True),
        patch("src.overlay.LimitedSets.retrieve_limited_sets", return_value=None),
        patch("src.overlay.AppUpdate.retrieve_file_version", return_value=("","")),
        patch("src.overlay.ArenaScanner", return_value=mock_scanner),
        patch("src.overlay.FileExtractor", return_value=None),
        patch("src.overlay.filter_options", return_value=["All Decks"]),
        patch("src.overlay.retrieve_arena_directory", return_value="fake_location"),
        patch("src.overlay.search_arena_log_locations", return_value="fake_location"),
    ):
        try:
            start_overlay()
        except Exception as e:
            pytest.fail(f"Exception occurred: {e}")

#TODO: create a test for CreateCardToolTip