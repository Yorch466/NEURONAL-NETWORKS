from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(224, 224, 3), num_regression_outputs=3, num_classes=3):
    base_model = EfficientNetB0(include_top=False, input_tensor=Input(shape=input_shape), weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Rama de regresión (altura, PB, PP)
    regression_output = Dense(num_regression_outputs, name='regression')(x)

    # Rama de clasificación (flaco, normal, sobrepeso)
    classification_output = Dense(num_classes, activation='softmax', name='classification')(x)

    model = Model(inputs=base_model.input, outputs=[regression_output, classification_output])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={
            'regression': 'mse',
            'classification': 'categorical_crossentropy'
        },
        metrics={
            'regression': 'mae',
            'classification': 'accuracy'
        }
    )

    return model
