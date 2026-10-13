# sine_project

Generated for esp32s3, ESP-IDF 5.5.1, esp-tflite-micro 1.4.1.

Build with `idf.py set-target esp32s3` and `idf.py build`.
Flash with `idf.py -p PORT flash`. Close the serial monitor before profiling.
Run `tinyml-deployer profile --port PORT --manifest deployment.json --report results.json`.

The firmware waits for a profiling request, performs a warm-up and then measures each provided sample repeatedly. Results include prediction agreement with the desktop TFLite reference. This checks implementation agreement, not task accuracy.

Inspect linker size reports and measured arena/heap usage before integration. The wrapper uses static storage and requires external serialization. Validation arrays consume flash. Remove the validation application when integrating the inference library into an application.
