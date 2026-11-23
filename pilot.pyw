"""Start the isolated RailSafe pilot profile."""

import multiprocessing
import os


if __name__ == "__main__":
    multiprocessing.freeze_support()
    os.environ["RAILSAFE_PILOT"] = "1"
    os.environ.pop("RAILSAFE_DB", None)

    from app.main import main

    main()
