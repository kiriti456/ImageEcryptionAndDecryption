import cv2
import numpy as np
import os

class ImageOperator:
    def __init__(self, image_path=None):
        self.image_path = image_path
        if image_path:
            self.original_image = self.load_image()

    def load_image(self):
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError("Image not found.")
        return image

    def grayscale_converter(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def convert_to_binary(self, image):
        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        return binary_image

    def decimal_converter(self, binary_image):
        decimal_image = binary_image.astype(np.uint8)
        return decimal_image

    def display_images(self, original, encrypted, decrypted):
        cv2.imshow('Original Image', original)
        cv2.imshow('Encrypted Image', encrypted.astype(np.uint8))
        cv2.imshow('Decrypted Image', decrypted.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class ImageEncryptionDecryption:
    def __init__(self):
        pass

    def sequence_generate_chaotic(self, rows, cols):
        chaotic_sequence = np.random.randint(0, 256, (rows, cols), dtype=np.uint8)
        return chaotic_sequence

    def image_scramble(self, image, chaotic_sequence):
        scrambled_image = cv2.bitwise_xor(image, chaotic_sequence)
        return scrambled_image

    def scramble_rubiks_cube(self, image, size_square):
        rows, cols = image.shape
        i, j = np.random.randint(0, rows - size_square + 1), np.random.randint(0, cols - size_square + 1)
        square = image[i:i + size_square, j:j + size_square]
        rotated_square = np.rot90(square, k=np.random.choice([1, 2, 3]), axes=(0, 1))
        image[i:i + size_square, j:j + size_square] = rotated_square
        return image

    def encrypt_image(self, original_image):
        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.threshold(grayscale_image, 128, 255, cv2.THRESH_BINARY)[1]

        rows, cols = binary_image.shape

        chaotic_sequence1 = self.sequence_generate_chaotic(rows, cols)
        chaotic_sequence2 = self.sequence_generate_chaotic(rows, cols)

        scrambled_image = self.image_scramble(binary_image, chaotic_sequence1)
        scrambled_image = self.image_scramble(scrambled_image, chaotic_sequence2)

        size_square = min(rows, cols) // 10 
        scrambled_image = self.scramble_rubiks_cube(scrambled_image, size_square)

        image_encrypt = scrambled_image.astype(np.uint8)

        return image_encrypt, chaotic_sequence1, chaotic_sequence2

    def decrypt_image(self, image_encrypt, chaotic_sequence1, chaotic_sequence2):
        image_decrypted = image_encrypt.copy()

        rows, cols = image_encrypt.shape

        size_square = min(rows, cols) // 10 

        if size_square >= 3:  
            for i in range(rows - size_square + 1):
                for j in range(cols - size_square + 1):
                    square = image_encrypt[i:i + size_square, j:j + size_square]
                    if np.all(square == image_encrypt[i:i + size_square, j:j + size_square]):
                        image_decrypted[i:i + size_square, j:j + size_square] = square

        image_decrypted = self.image_scramble(image_decrypted, chaotic_sequence2)
        image_decrypted = self.image_scramble(image_decrypted, chaotic_sequence1)

        return image_decrypted

def process_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError("Folder not found.")
    
    output_folder = os.path.join(folder_path, 'processed_images')
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        try:
            image_operator = ImageOperator(image_path)
            image_encryption_decryption = ImageEncryptionDecryption()

            image_encrypt, key1, key2 = image_encryption_decryption.encrypt_image(image_operator.original_image)

            image_decrypted = image_encryption_decryption.decrypt_image(image_encrypt, key1, key2)

            output_file_encrypted = os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}_encrypted.png')
            output_file_decrypted = os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}_decrypted.png')

            cv2.imwrite(output_file_encrypted, image_encrypt)
            cv2.imwrite(output_file_decrypted, image_decrypted)

            print(f"Processed: {image_file}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    folder_path = input('Please provide the input folder path to encrypt and decrypt the images : ')

    try:
        process_images_in_folder(folder_path)

    except Exception as e:
        print(f"Error: {e}")
