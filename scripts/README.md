# scripts/

Utility scripts for the tinyml-deployer project.

## record_demo.py

Runs a live-style demo of the CLI commands with pauses between them. Intended
to be recorded as a terminal screencast.

### Recording with asciinema

1. Install asciinema:

   ```bash
   pip install asciinema
   ```

2. Record the demo:

   ```bash
   asciinema rec demo.cast -c "python scripts/record_demo.py"
   ```

3. Convert to GIF (requires [agg](https://github.com/asciinema/agg)):

   ```bash
   agg demo.cast demo.gif
   ```

   Or upload directly to asciinema.org:

   ```bash
   asciinema upload demo.cast
   ```
