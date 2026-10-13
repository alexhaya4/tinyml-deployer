idf_component_register(SRCS "main.cpp" "model_data.cpp" "inference.cpp"
  INCLUDE_DIRS "." REQUIRES esp_timer esp_system app_update)
