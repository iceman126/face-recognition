from pathlib import Path
from PIL import Image
import numpy as np
from matplotlib import pyplot
import dlib

detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_embedding(model, face_pixels):
    # face_pixels = face_pixels.astype('float32')
    # mean, std = face_pixels.mean(), face_pixels.std()
    # face_pixels = (face_pixels - mean) / std
    # print(face_pixels.shape)
    # samples = np.expand_dims(face_pixels, axis=0)
    embedding = model.compute_face_descriptor(face_pixels)
    return embedding

def extract_face(img_file, required_size=(150, 150)):
    image = Image.open(img_file)
    image = image.convert("RGB")
    pixels = np.asarray(image)
    results = detector(pixels, 2)       # need to upsample 2 times, or some faces cannot be detected
    left, top, width, height = results[0].rect.left(), results[0].rect.top(), results[0].rect.width(), results[0].rect.height()
    face = pixels[top:top+height, left:left+width]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def load_faces(directory):
    faces = list()
    for f in directory.iterdir():
        face = extract_face(str(f))
        faces.append(face)
    return faces

def load_dataset(directory):
    X, y = list(), list()
    for subdir in directory.iterdir():
        if subdir.is_dir() == False:
            continue
        faces = load_faces(subdir)
        labels = [subdir.name for _ in range(len(faces))]
        print(f"Loaded {len(faces)} examples for class: {subdir.name}")
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

def main():
    # load dataset and extract faces
    train_X, train_y = load_dataset(Path("5_celebrity/train"))
    print(train_X.shape, train_y.shape)

    val_X, val_y = load_dataset(Path("5_celebrity/val"))
    print(val_X.shape, val_y.shape)

    # uncomment this section if intermediate faces output are needed
    # np.savez_compressed('5-celebrity-faces-dataset.npz', train_X, train_y, val_X, val_y)
    # print("Save face embeddings to 5-celebrity-faces-dataset.npz")

    model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    print("Loaded model from dlib_face_recognition_resnet_model_v1.dat")

    # convert faces to embeddings
    embed_train_X = list()
    for face_pixels in train_X:
        embedding = get_embedding(model, face_pixels)
        embed_train_X.append(embedding)
    embed_train_X = np.asarray(embed_train_X)
    print(embed_train_X.shape)

    embed_val_X = list()
    for face_pixels in val_X:
        embedding = get_embedding(model, face_pixels)
        embed_val_X.append(embedding)
    embed_val_X = np.asarray(embed_val_X)
    print(embed_val_X.shape)

    np.savez_compressed('5-celebrity-faces-embeddings.npz', embed_train_X, train_y, embed_val_X, val_y)
    print(f"Saved faces embeddings: {train_X.shape}, {train_y.shape}, {embed_val_X.shape}, {val_y.shape}")

if __name__ == "__main__":
    main()