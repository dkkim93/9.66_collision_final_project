import time
import pickle
import cv2
from util import read_filename


def write_text(image, text, loc):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.0
    color = (0, 0, 0)  # Black
    thickness = 1
    lineType = cv2.LINE_AA

    cv2.putText(
        img=image,
        text=text,
        org=loc,
        fontFace=font,
        fontScale=fontScale,
        color=color,
        thickness=thickness,
        lineType=lineType)


if __name__ == "__main__":
    # Read filenames in directory
    filename_n = read_filename(path="./data")

    # Show image and receive user input
    for filename in filename_n:
        user_input = ""

        while True:
            image = cv2.imread(filename)
            image = cv2.resize(image, (600, 600))
            write_text(image, text="Response: ", loc=(60, 50))
            write_text(image, text=user_input, loc=(250, 50))
            cv2.imshow("image", image)

            c = cv2.waitKey(1)
            # NOTE -1 is when no key is pressed
            if c != -1:
                # If enter is pressed then save data
                # and move to next image
                if ord(chr(c)) == 13:
                    try:
                        if float(user_input) < 0 or float(user_input) > 10:
                            write_text(image, text="Invalid Input", loc=(60, 150))
                            time.sleep(0.2)
                        else:
                            break
                    except ValueError:
                        write_text(image, text="Invalid Input", loc=(60, 150))
                        time.sleep(0.2)
                else:
                    # Shift is pressed
                    if ord(chr(c)) == 255:
                        pass

                    # Backspace is pressed
                    elif ord(chr(c)) == 8:
                        user_input = user_input[:-1]

                    else:
                        user_input += chr(c)

        # Save data with label (remove .png)
        with open(filename[:-4] + "_label", "wb") as output:
            pickle.dump(user_input, output, pickle.HIGHEST_PROTOCOL)
