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
            write_text(image, text="Enter 1 for collision", loc=(60, 50))
            write_text(image, text="Enter 0 for no-collision", loc=(60, 100))
            write_text(image, text="Response: ", loc=(60, 150))
            write_text(image, text=user_input, loc=(250, 150))
            cv2.imshow("image", image)

            c = cv2.waitKey(1)
            # NOTE -1 is when no key is pressed
            if c != -1:
                # If enter is pressed then save data
                # and move to next image
                if ord(chr(c)) == 13:
                    break
                else:
                    # Backspace is pressed
                    if ord(chr(c)) == 8:
                        user_input = user_input[:-1]
                    else:
                        user_input += chr(c)
