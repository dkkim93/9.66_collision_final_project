import cv2
import time
import pickle


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
    # Set parameters
    IMG_PATH = "../data/imgs/"
    N_EPISODE = 89
    MAX_TIME = 5

    # Show image and receive user input
    for i_episode in range(N_EPISODE):
        user_input = ""
        i_time = 0

        while True:
            filename = IMG_PATH + "course_966_e_" + "%02d" % (i_episode) + "_t_" + "%02d" % (i_time) + ".png"

            image = cv2.imread(filename)
            write_text(image, text="Response: ", loc=(60, 50))
            write_text(image, text=user_input, loc=(250, 50))
            cv2.imshow("image", image)

            c = cv2.waitKey(2000)  # Wait for x ms
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
                    # Right arrow is pressed
                    elif ord(chr(c)) == 83:
                        i_time += 1
                        if i_time > MAX_TIME:
                            i_time = MAX_TIME
                    elif ord(chr(c)) == 81:
                        i_time -= 1
                        if i_time < 0:
                            i_time = 0
                    else:
                        user_input += chr(c)

        # Save data with label (remove .png)
        with open(filename[:-4] + "_label", "wb") as output:
            pickle.dump(user_input, output, pickle.HIGHEST_PROTOCOL)
