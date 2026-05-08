from pathlib import Path
import shutil
import subprocess
import tempfile
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk
import tifffile as tiff


DEFAULT_CLASSES = ("bacteria", "background", "artifacts", "uncertain", "ignore")
AXES = "TZCYX"
DISPLAY_MAX_VALUE = 1500
SCREEN_WIDTH_FRACTION = 0.9
SCREEN_HEIGHT_FRACTION = 0.75
PREVIEW_MAX_HEIGHT = 1800
DEFAULT_DISPLAY_CHANNELS = (0,)
DEFAULT_FIJI_EXECUTABLE_NAMES = ("ImageJ-linux64", "imagej", "fiji")
WINDOW_WIDTH_FRACTION = 0.7
WINDOW_HEIGHT_FRACTION = 0.8
MIN_WINDOW_WIDTH = 1100
MIN_WINDOW_HEIGHT = 900
PREVIEW_UPSCALE_FACTOR = 3


class EventGui(tk.Tk):
    def __init__(
        self,
        event,
        event_name,
        main_folder,
        class_names=DEFAULT_CLASSES,
        display_channels=DEFAULT_DISPLAY_CHANNELS,
        fiji_executable=None,
    ):
        super().__init__()

        self.event = event
        self.event_name = event_name
        self.main_folder = Path(main_folder)
        self.class_names = class_names
        self.display_channels = tuple(display_channels)
        self.fiji_executable = fiji_executable
        self.preview_image = None
        self.saved_class_name = None

        self.title(f"Classify event: {self.event_name}")
        self.event_screen()
        self.configure_window()

    def event_screen(self):
        preview = self.make_preview_image(self.event)
        self.preview_image = ImageTk.PhotoImage(preview)

        preview_frame = tk.Frame(self)
        preview_frame.pack(padx=12, pady=12, fill="x")

        canvas_width = min(
            preview.width,
            int(self.winfo_screenwidth() * SCREEN_WIDTH_FRACTION),
        )
        canvas_height = preview.height

        self.image_canvas = tk.Canvas(
            preview_frame,
            width=canvas_width,
            height=canvas_height,
            highlightthickness=0,
            xscrollincrement=32,
        )
        horizontal_scrollbar = tk.Scrollbar(
            preview_frame,
            orient="horizontal",
            command=self.image_canvas.xview,
        )
        self.image_canvas.configure(xscrollcommand=horizontal_scrollbar.set)
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.preview_image)
        self.image_canvas.configure(scrollregion=(0, 0, preview.width, preview.height))

        self.image_canvas.pack(side="top", fill="x")
        horizontal_scrollbar.pack(side="top", fill="x")
        self.image_canvas.bind("<MouseWheel>", self.scroll_preview_horizontally)
        self.image_canvas.bind("<Button-4>", self.scroll_preview_horizontally)
        self.image_canvas.bind("<Button-5>", self.scroll_preview_horizontally)

        button_frame = tk.Frame(self)
        button_frame.pack(padx=12, pady=(0, 12), fill="x")

        inspect_button = tk.Button(
            button_frame,
            text="Open in Fiji",
            command=self.open_in_fiji,
        )
        inspect_button.pack(side="left", padx=4)

        for class_name in self.class_names:
            folder = self.main_folder / class_name
            if class_name in ("ignore", "discard"):
                button = IgnoreButton(button_frame, class_name)
            else:
                button = SaveFolderButton(button_frame, class_name, folder)
            button.pack(side="left", expand=True, fill="x", padx=4)

        self.status_label = tk.Label(self, text="Choose a class or ignore this event.")
        self.status_label.pack(padx=12, pady=(0, 12))

    def configure_window(self):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        target_width = max(int(screen_width * WINDOW_WIDTH_FRACTION), MIN_WINDOW_WIDTH)
        target_height = max(int(screen_height * WINDOW_HEIGHT_FRACTION), MIN_WINDOW_HEIGHT)

        target_width = min(target_width, screen_width)
        target_height = min(target_height, screen_height)

        self.geometry(f"{target_width}x{target_height}")
        self.update_idletasks()

        actual_width = self.winfo_width()
        actual_height = self.winfo_height()

        pos_x = max((screen_width - actual_width) // 2, 0)
        pos_y = max((screen_height - actual_height) // 2, 0)

        self.geometry(f"{actual_width}x{actual_height}+{pos_x}+{pos_y}")
        self.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

    def make_preview_image(self, event):
        image = np.asarray(event)
        image = np.squeeze(image)

        if image.ndim == 2:
            preview = self.to_uint8_preview(image)
        elif image.ndim == 3:
            preview = self.tile_channels(image)
        elif image.ndim == 4:
            preview = self.tile_z_stack_channels(image)
        else:
            raise ValueError(
                "Expected event with shape [Y, X], [C, Y, X], or [Z, C, Y, X], "
                f"got {tuple(image.shape)}."
            )

        preview_image = Image.fromarray(preview)
        return self.scale_preview_to_screen(preview_image)

    def scale_preview_to_screen(self, image):
        image = self.upscale_small_preview(image)

        max_height = min(
            int(self.winfo_screenheight() * SCREEN_HEIGHT_FRACTION),
            PREVIEW_MAX_HEIGHT,
        )

        if image.height > max_height:
            target_width = max(1, int(image.width * (max_height / image.height)))
            image = image.resize((target_width, max_height), Image.Resampling.NEAREST)

        return image

    def scroll_preview_horizontally(self, event):
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1

        self.image_canvas.xview_scroll(delta * 3, "units")
        return "break"

    def upscale_small_preview(self, image):
        width, height = image.size
        upscaled_size = (
            width * PREVIEW_UPSCALE_FACTOR,
            height * PREVIEW_UPSCALE_FACTOR,
        )
        return image.resize(upscaled_size, Image.Resampling.NEAREST)

    def get_display_channels(self, num_channels):
        channels = self.display_channels or DEFAULT_DISPLAY_CHANNELS
        valid_channels = [channel for channel in channels if 0 <= channel < num_channels]

        if not valid_channels:
            raise ValueError(
                f"No valid display channels found for event with {num_channels} channels. "
                f"Requested channels: {channels}."
            )

        return valid_channels

    def tile_channels(self, image):
        channels = self.get_display_channels(image.shape[0])
        tiles = [self.to_uint8_preview(image[channel]) for channel in channels]
        return np.concatenate(tiles, axis=0)

    def tile_z_stack_channels(self, image):
        z_slices, num_channels = image.shape[:2]
        channels = self.get_display_channels(num_channels)

        channel_rows = []
        for channel in channels:
            row_tiles = [
                self.to_uint8_preview(image[z_index, channel])
                for z_index in range(z_slices)
            ]
            channel_rows.append(np.concatenate(row_tiles, axis=1))

        return np.concatenate(channel_rows, axis=0)

    def to_uint8_preview(self, image, max_value=DISPLAY_MAX_VALUE):
        image = image.astype(np.float32)
        image = np.clip(image, 0, max_value)
        if max_value > 0:
            image /= max_value

        return (image * 255).astype(np.uint8)

    def get_event_axes(self):
        return AXES[-self.event.ndim:]

    def write_event_tiff(self, output_path):
        tiff.imwrite(
            output_path,
            self.event,
            imagej=True,
            metadata={
                "mode": "color",
                "axes": self.get_event_axes(),
                "unit": "micron",
                "spacing": 0.5,
            },
            resolution=(5.884696082960086, 5.884696082960086),
        )

    def resolve_fiji_executable(self):
        if self.fiji_executable:
            candidate = Path(self.fiji_executable).expanduser()
            if candidate.exists():
                return str(candidate)
            resolved = shutil.which(self.fiji_executable)
            if resolved is not None:
                return resolved
            return None

        for executable_name in DEFAULT_FIJI_EXECUTABLE_NAMES:
            resolved = shutil.which(executable_name)
            if resolved is not None:
                return resolved

        return None

    def open_in_fiji(self):
        executable = self.resolve_fiji_executable()
        if executable is None:
            self.status_label.config(
                text="Fiji executable not found. Set fiji_executable in the script."
            )
            return

        temp_dir = Path(tempfile.gettempdir()) / "mcrop_events_gui_fiji"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{self.event_name}.tif"
        self.write_event_tiff(temp_path)

        try:
            subprocess.Popen([executable, str(temp_path)])
        except OSError as exc:
            self.status_label.config(text=f"Could not open Fiji: {exc}")
            return

        self.status_label.config(text=f"Opened in Fiji: {temp_path.name}")

    def save_current_event(self, folder, class_name):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        output_path = folder / f"{self.event_name}.tif"
        self.write_event_tiff(output_path)

        self.saved_class_name = class_name
        self.status_label.config(text=f"Saved to {folder.name}")
        self.destroy()

    def ignore_current_event(self, class_name="ignore"):
        self.saved_class_name = class_name
        self.destroy()


class SaveFolderButton(tk.Button):
    def __init__(self, parent, class_name, folder):
        self.class_name = class_name
        self.folder = Path(folder)

        super().__init__(
            parent,
            text=class_name,
            command=self.save_event,
        )

    def save_event(self):
        app = self.winfo_toplevel()
        app.save_current_event(self.folder, self.class_name)


class IgnoreButton(tk.Button):
    def __init__(self, parent, class_name="ignore"):
        self.class_name = class_name

        super().__init__(
            parent,
            text="ignore",
            command=self.ignore_event,
        )

    def ignore_event(self):
        app = self.winfo_toplevel()
        app.ignore_current_event(self.class_name)


class EventBrowserGui(EventGui):
    def __init__(
        self,
        event_records_by_image,
        get_event_context,
        mark_saved,
        main_folder,
        class_names=DEFAULT_CLASSES,
        display_channels=DEFAULT_DISPLAY_CHANNELS,
        fiji_executable=None,
    ):
        tk.Tk.__init__(self)

        self.event_records_by_image = event_records_by_image
        self.get_event_context = get_event_context
        self.mark_saved = mark_saved
        self.main_folder = Path(main_folder)
        self.class_names = class_names
        self.display_channels = tuple(display_channels)
        self.fiji_executable = fiji_executable
        self.preview_image = None
        self.saved_class_name = None
        self.event = None
        self.event_name = None
        self.current_bounding_box = None
        self.current_already_saved = False
        self.current_saved_message = ""
        self.current_overlap_message = ""
        self.image_names = list(event_records_by_image)

        self.title("Classify events")
        self.event_screen()
        self.configure_window()

    def event_screen(self):
        controls_frame = tk.Frame(self)
        controls_frame.pack(padx=12, pady=(12, 6), fill="x")

        tk.Label(controls_frame, text="Image").pack(side="left", padx=(0, 4))
        self.image_combobox = ttk.Combobox(
            controls_frame,
            values=self.image_names,
            state="readonly",
            width=48,
        )
        self.image_combobox.pack(side="left", padx=(0, 12), fill="x", expand=True)
        self.image_combobox.bind("<<ComboboxSelected>>", self.on_image_selected)

        tk.Label(controls_frame, text="Event").pack(side="left", padx=(0, 4))
        self.previous_event_button = tk.Button(
            controls_frame,
            text="<",
            command=self.previous_event,
        )
        self.previous_event_button.pack(side="left", padx=(0, 4))

        self.event_combobox = ttk.Combobox(
            controls_frame,
            state="readonly",
            width=36,
        )
        self.event_combobox.pack(side="left", fill="x", expand=True)
        self.event_combobox.bind("<<ComboboxSelected>>", self.on_event_selected)

        self.next_event_button = tk.Button(
            controls_frame,
            text=">",
            command=self.next_event,
        )
        self.next_event_button.pack(side="left", padx=(4, 0))

        preview_frame = tk.Frame(self)
        preview_frame.pack(padx=12, pady=6, fill="x")

        self.image_canvas = tk.Canvas(
            preview_frame,
            width=int(self.winfo_screenwidth() * SCREEN_WIDTH_FRACTION),
            height=int(self.winfo_screenheight() * SCREEN_HEIGHT_FRACTION),
            highlightthickness=0,
            xscrollincrement=32,
        )
        horizontal_scrollbar = tk.Scrollbar(
            preview_frame,
            orient="horizontal",
            command=self.image_canvas.xview,
        )
        self.image_canvas.configure(xscrollcommand=horizontal_scrollbar.set)
        self.image_canvas.pack(side="top", fill="x")
        horizontal_scrollbar.pack(side="top", fill="x")
        self.image_canvas.bind("<MouseWheel>", self.scroll_preview_horizontally)
        self.image_canvas.bind("<Button-4>", self.scroll_preview_horizontally)
        self.image_canvas.bind("<Button-5>", self.scroll_preview_horizontally)

        button_frame = tk.Frame(self)
        button_frame.pack(padx=12, pady=(0, 12), fill="x")

        inspect_button = tk.Button(
            button_frame,
            text="Open in Fiji",
            command=self.open_in_fiji,
        )
        inspect_button.pack(side="left", padx=4)

        for class_name in self.class_names:
            folder = self.main_folder / class_name
            if class_name in ("ignore", "discard"):
                button = IgnoreButton(button_frame, class_name)
            else:
                button = SaveFolderButton(button_frame, class_name, folder)
            button.pack(side="left", expand=True, fill="x", padx=4)

        self.status_label = tk.Label(self, text="Choose an image and event.")
        self.status_label.pack(padx=12, pady=(0, 12), fill="x")

        if self.image_names:
            self.image_combobox.current(0)
            self.on_image_selected()

    def on_image_selected(self, event=None):
        image_name = self.image_combobox.get()
        event_labels = [
            event_record["label"]
            for event_record in self.event_records_by_image[image_name]
        ]
        self.event_combobox.configure(values=event_labels)

        if event_labels:
            self.event_combobox.current(0)
            self.display_selected_event()
        else:
            self.update_event_navigation_buttons()

    def on_event_selected(self, event=None):
        self.display_selected_event()

    def previous_event(self):
        event_index = self.event_combobox.current()

        if event_index > 0:
            self.event_combobox.current(event_index - 1)
            self.display_selected_event()

    def next_event(self):
        event_index = self.event_combobox.current()
        event_count = len(self.event_combobox["values"])

        if event_index + 1 < event_count:
            self.event_combobox.current(event_index + 1)
            self.display_selected_event()

    def display_selected_event(self):
        image_name = self.image_combobox.get()
        event_index = self.event_combobox.current()

        if not image_name or event_index < 0:
            return

        context = self.get_event_context(image_name, event_index)
        self.event = context["event"]
        self.event_name = context["event_name"]
        self.current_bounding_box = context["bounding_box"]
        self.current_already_saved = context["already_saved"]
        self.current_saved_message = context["saved_message"]
        self.current_overlap_message = context["overlap_message"]
        self.saved_class_name = None

        preview = self.make_preview_image(self.event)
        self.preview_image = ImageTk.PhotoImage(preview)

        canvas_width = min(
            preview.width,
            int(self.winfo_screenwidth() * SCREEN_WIDTH_FRACTION),
        )
        self.image_canvas.configure(
            width=canvas_width,
            height=preview.height,
            scrollregion=(0, 0, preview.width, preview.height),
        )
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.preview_image)
        self.image_canvas.xview_moveto(0)

        self.update_status("Choose a class or ignore this event.")
        self.update_event_navigation_buttons()

    def save_current_event(self, folder, class_name):
        if self.event is None:
            return

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        output_path = folder / f"{self.event_name}.tif"
        self.write_event_tiff(output_path)

        self.saved_class_name = class_name
        if not self.current_already_saved:
            self.mark_saved(
                self.image_combobox.get(),
                self.event_combobox.current(),
                self.current_bounding_box,
                class_name,
            )
        self.update_status(f"Saved to {folder.name}")
        self.go_to_next_event()

    def ignore_current_event(self, class_name="ignore"):
        self.saved_class_name = class_name
        self.update_status("Ignored event.")
        self.go_to_next_event()

    def go_to_next_event(self):
        event_index = self.event_combobox.current()
        event_count = len(self.event_combobox["values"])

        if event_index + 1 < event_count:
            self.event_combobox.current(event_index + 1)
            self.display_selected_event()
            return

        image_index = self.image_combobox.current()
        if image_index + 1 < len(self.image_names):
            self.image_combobox.current(image_index + 1)
            self.on_image_selected()
            return

        self.update_status("No more events.")

    def update_status(self, action_message):
        messages = [action_message]
        if self.current_saved_message:
            messages.append(self.current_saved_message)
        if self.current_overlap_message:
            messages.append(self.current_overlap_message)

        self.status_label.config(text=" ".join(messages))

    def update_event_navigation_buttons(self):
        event_index = self.event_combobox.current()
        event_count = len(self.event_combobox["values"])

        previous_state = "normal" if event_index > 0 else "disabled"
        next_state = "normal" if event_index + 1 < event_count else "disabled"

        self.previous_event_button.config(state=previous_state)
        self.next_event_button.config(state=next_state)
