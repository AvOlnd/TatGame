import cv2
import time
import random
import numpy as np

from ultralytics import YOLO
from tasks import TASKS


# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_PATH = "yolo11n.pt"

CAMERA_ID = 0

SQUARE_SIZE = 350

CONFIDENCE = 0.45

SUCCESS_MESSAGE_TIME = 1.2

NUMBER_OF_TASKS = 10

WINDOW_NAME = "Tatar YOLO Game"


# ============================================================
# ИГНОРИРУЕМЫЕ КЛАССЫ
# ============================================================

# Эти объекты YOLO полностью игнорирует.
# Они не будут участвовать в проверке ответа
# и не будут рисоваться на экране.

IGNORED_CLASSES = {
    "person",
}


# ============================================================
# ЦВЕТА
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GREEN = (0, 220, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 220)


# ============================================================
# ЗАГРУЗКА YOLO
# ============================================================

print("YOLO моделе йөкләнә...")

model = YOLO(MODEL_PATH)

print("YOLO әзер!")


# ============================================================
# КАМЕРА
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("Камераны ачып булмады!")
    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# ПОДГОТОВКА ЗАДАНИЙ
# ============================================================

tasks = TASKS.copy()

random.shuffle(tasks)

if NUMBER_OF_TASKS > len(tasks):
    NUMBER_OF_TASKS = len(tasks)

tasks = tasks[:NUMBER_OF_TASKS]


current_task_index = 0
score = 0

game_finished = False

success_message = ""
success_message_start = 0


# ============================================================
# ФУНКЦИИ
# ============================================================

def get_center_square(frame):

    height, width = frame.shape[:2]

    size = min(
        SQUARE_SIZE,
        width - 40,
        height - 100
    )

    x1 = int(width / 2 - size / 2)
    y1 = int(height / 2 - size / 2)

    x2 = int(width / 2 + size / 2)
    y2 = int(height / 2 + size / 2)

    return x1, y1, x2, y2


def box_center(box):

    x1, y1, x2, y2 = box

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    return cx, cy


def is_point_inside_square(point, square):

    px, py = point

    x1, y1, x2, y2 = square

    return (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    )


def draw_center_square(
    frame,
    square,
    correct=False
):

    x1, y1, x2, y2 = square

    if correct:
        color = GREEN
    else:
        color = WHITE

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        4
    )


def draw_text(
    frame,
    text,
    position,
    size=1,
    color=WHITE,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_header(
    frame,
    task_text,
    score,
    task_number,
    total_tasks
):

    height, width = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, 115),
        BLACK,
        -1
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )

    draw_text(
        frame,
        task_text,
        (30, 45),
        size=1.1,
        color=WHITE,
        thickness=2
    )

    draw_text(
        frame,
        f"Score: {score}",
        (30, 85),
        size=0.8,
        color=YELLOW,
        thickness=2
    )

    task_number_text = (
        f"{task_number}/{total_tasks}"
    )

    text_size = cv2.getTextSize(
        task_number_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        2
    )[0]

    draw_text(
        frame,
        task_number_text,
        (
            width - text_size[0] - 30,
            45
        ),
        size=0.8,
        color=WHITE,
        thickness=2
    )


def draw_message(
    frame,
    text,
    color
):

    height, width = frame.shape[:2]

    font_scale = 1.5
    thickness = 4

    text_size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )[0]

    x = int(
        (width - text_size[0]) / 2
    )

    y = int(
        height * 0.85
    )

    padding = 25

    cv2.rectangle(
        frame,
        (
            x - padding,
            y - text_size[1] - padding
        ),
        (
            x + text_size[0] + padding,
            y + padding
        ),
        BLACK,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_finish_screen(
    frame,
    score,
    total
):

    height, width = frame.shape[:2]

    overlay = np.zeros_like(frame)

    frame[:] = cv2.addWeighted(
        frame,
        0.25,
        overlay,
        0.75,
        0
    )

    title = "Уен тәмам!"

    title_size = cv2.getTextSize(
        title,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.7,
        4
    )[0]

    title_x = int(
        (width - title_size[0]) / 2
    )

    cv2.putText(
        frame,
        title,
        (
            title_x,
            int(height * 0.35)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.7,
        WHITE,
        4,
        cv2.LINE_AA
    )

    result = (
        f"Нәтиҗә: {score}/{total}"
    )

    result_size = cv2.getTextSize(
        result,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        3
    )[0]

    result_x = int(
        (width - result_size[0]) / 2
    )

    cv2.putText(
        frame,
        result,
        (
            result_x,
            int(height * 0.50)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        YELLOW,
        3,
        cv2.LINE_AA
    )

    restart = "R - яңадан башларга"

    restart_size = cv2.getTextSize(
        restart,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        2
    )[0]

    restart_x = int(
        (width - restart_size[0]) / 2
    )

    cv2.putText(
        frame,
        restart,
        (
            restart_x,
            int(height * 0.65)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        WHITE,
        2,
        cv2.LINE_AA
    )

    exit_text = "Q - чыгу"

    exit_size = cv2.getTextSize(
        exit_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        2
    )[0]

    exit_x = int(
        (width - exit_size[0]) / 2
    )

    cv2.putText(
        frame,
        exit_text,
        (
            exit_x,
            int(height * 0.72)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        WHITE,
        2,
        cv2.LINE_AA
    )


def restart_game():

    global tasks
    global current_task_index
    global score
    global game_finished
    global success_message

    tasks = TASKS.copy()

    random.shuffle(tasks)

    number = min(
        NUMBER_OF_TASKS,
        len(tasks)
    )

    tasks = tasks[:number]

    current_task_index = 0

    score = 0

    game_finished = False

    success_message = ""


# ============================================================
# ОСНОВНОЙ ЦИКЛ
# ============================================================

print()
print("Уен башланды!")
print("Q - чыгу")
print("R - яңадан башлау")
print()


while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Камерадан кадр алу мөмкин түгел."
        )

        break


    # Зеркальное отображение
    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # ЭКРАН ОКОНЧАНИЯ
    # ========================================================

    if game_finished:

        draw_finish_screen(
            frame,
            score,
            len(tasks)
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            restart_game()

        continue


    # ========================================================
    # ТЕКУЩЕЕ ЗАДАНИЕ
    # ========================================================

    current_task = (
        tasks[current_task_index]
    )

    required_class = (
        current_task["class_name"]
    )

    task_text = (
        current_task["text"]
    )


    # ========================================================
    # ЦЕНТРАЛЬНЫЙ КВАДРАТ
    # ========================================================

    square = get_center_square(
        frame
    )


    # ========================================================
    # YOLO
    # ========================================================

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )


    correct_object_inside = False

    detected_objects = []


    # ========================================================
    # ОБРАБОТКА YOLO
    # ========================================================

    for result in results:

        boxes = result.boxes

        for box in boxes:

            # ------------------------------------------------
            # Координаты объекта
            # ------------------------------------------------

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = map(
                int,
                xyxy
            )


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = float(
                box.conf[0]
                .cpu()
                .numpy()
            )


            # ------------------------------------------------
            # ID класса
            # ------------------------------------------------

            class_id = int(
                box.cls[0]
                .cpu()
                .numpy()
            )


            # ------------------------------------------------
            # Название класса
            # ------------------------------------------------

            class_name = (
                model.names[class_id]
            )


            # =================================================
            # ИГНОРИРУЕМ ЛЮДЕЙ И ТЕЛЕФОНЫ
            # =================================================

            if class_name in IGNORED_CLASSES:

                continue


            # ------------------------------------------------
            # Центр объекта
            # ------------------------------------------------

            cx, cy = box_center(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


            # ------------------------------------------------
            # Проверяем квадрат
            # ------------------------------------------------

            inside = (
                is_point_inside_square(
                    (cx, cy),
                    square
                )
            )


            # ------------------------------------------------
            # Сохраняем объект
            # ------------------------------------------------

            detected_objects.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "center": (
                        cx,
                        cy
                    ),
                    "inside": inside
                }
            )


    # ========================================================
    # РИСУЕМ ОБЪЕКТЫ
    # ========================================================

    for obj in detected_objects:

        class_name = (
            obj["class_name"]
        )

        confidence = (
            obj["confidence"]
        )

        x1, y1, x2, y2 = (
            obj["box"]
        )

        cx, cy = (
            obj["center"]
        )

        inside = (
            obj["inside"]
        )


        # ----------------------------------------------------
        # Проверяем, является ли объект нужным
        # ----------------------------------------------------

        is_required = (
            class_name == required_class
        )


        # ----------------------------------------------------
        # Цвет bounding box
        # ----------------------------------------------------

        if is_required and inside:

            color = GREEN

            correct_object_inside = True

        elif inside:

            color = RED

        else:

            color = WHITE


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        # ----------------------------------------------------
        # Центр объекта
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (cx, cy),
            5,
            color,
            -1
        )


        # ----------------------------------------------------
        # Название объекта
        # ----------------------------------------------------

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(y1 - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # РИСУЕМ ЦЕНТРАЛЬНЫЙ КВАДРАТ
    # ========================================================

    draw_center_square(
        frame,
        square,
        correct_object_inside
    )


    # ========================================================
    # ВЕРХНИЙ ИНТЕРФЕЙС
    # ========================================================

    draw_header(
        frame,
        task_text,
        score,
        current_task_index + 1,
        len(tasks)
    )


    # ========================================================
    # ПРОВЕРКА ОТВЕТА
    # ========================================================

    current_time = time.time()


    if correct_object_inside:

        # ----------------------------------------------------
        # Первое обнаружение правильного предмета
        # ----------------------------------------------------

        if success_message == "":

            success_message = "ДӨРЕС! ✓"

            success_message_start = (
                current_time
            )

            score += 1


        # ----------------------------------------------------
        # Показываем сообщение
        # ----------------------------------------------------

        draw_message(
            frame,
            success_message,
            GREEN
        )


        # ----------------------------------------------------
        # Переходим к следующему заданию
        # ----------------------------------------------------

        if (
            current_time
            - success_message_start
            >= SUCCESS_MESSAGE_TIME
        ):

            success_message = ""

            current_task_index += 1


            # ------------------------------------------------
            # Все задания выполнены
            # ------------------------------------------------

            if (
                current_task_index
                >= len(tasks)
            ):

                game_finished = True


    # ========================================================
    # ПОКАЗ КАДРА
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    # КЛАВИАТУРА
    # ========================================================

    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # Q — выход

    if key == ord("q"):

        break


    # ESC — выход

    if key == 27:

        break


    # R — новая игра

    if key == ord("r"):

        restart_game()


# ============================================================
# ЗАКРЫТИЕ
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print("Уен тәмам.")
print(
    f"Нәтиҗә: {score}/{len(tasks)}"
)