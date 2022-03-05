def Load Yolo_nodel():
    gpus = tf.config.experimental.list_physical_devices( 'GPU')
   if len(gpus) > 0:
       print(f'GPUS {gpus}')
        try: tf.config.experimental.set_menory_growth(gpus[0], True)
        except RuntineError: pass
    if YOLO_FRAMEWORK = "tf": # TensorFlow detection
        if YOLO_TYPE == "yolov4":
            Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
        if YOLO_TYPE == "yolov3":
            Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS
        if YOLO_CUSTOM_WEIGHTS == False:
            print("Loading Darknet_weights from:", Darknet_weights)
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_cOco_CLASSES)
            load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
        else:
            checkpoint = f"./checkpoints/{TRAIN_MODEL_NAME}"
            if TRAIN_YOLO_TINY:
                checkpoint += "_Tiny"
            print("Loading custon weights from:", checkpoint)
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
            yolo.load_weights(checkpoint) # use custon weights
    elif YOLO_FRAMEWORK == "trt": # TensorRT detection
        saved_nodel_loaded = tf.saved_nodel.load(YOLO_CUSTOM_WEIGHTS, tags=[tag_constants.SERVING])
        signature_keys = list(saved_nodel_loaded.signatures.keys())
       yolo = saved_nodel_loaded.signatures['serving_default']
    return yolo
