from MCrop_events_gui.gui import EventGui
from utils import read_image, get_images_path

images_path = get_images_path("/data/joao/Materials/Joao_repository/Dataset/0unsure")

for path in images_path:
    event = read_image(path)

    app = EventGui(
        event=event,
        event_name=path.stem,
        main_folder="/data/joao/Materials/Joao_repository/TEST_GUI",
        class_names=("bacteria", "background", "artifacts", "uncertain", "ignore"),
    )
    app.mainloop()
