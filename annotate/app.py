import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import re
import time
from skimage import io
import os
import cv2

DEBUG = True
NUM_ATYPES = 15
DEFAULT_FIG_MODE = "layout"

# read images
images_name = 'my_id.jpg'
images_main_dir = 'images/'
images_folders = os.listdir(images_main_dir)
filelist = list(map(lambda p: images_main_dir + p + '/' + images_name, images_folders))

annotation_colormap = px.colors.qualitative.Light24
annotation_types = [
    'first_name',
    'last_name',
    'id'
]


ANN_DIR_PATH = "annotations/"

if os.path.exists(ANN_DIR_PATH):
    annotataed_files = os.listdir(ANN_DIR_PATH)
else:
    annotataed_files = []

annotations_class = {'first_name': 0, 'last_name': 1, 'id': 2}
DEFAULT_ATYPE = annotation_types[0]

# prepare bijective type<->color mapping
typ_col_pairs = [
    (t, annotation_colormap[n % len(annotation_colormap)])
    for n, t in enumerate(annotation_types)
]

# types to colors
color_dict = {}

# colors to types
type_dict = {}
for typ, col in typ_col_pairs:
    color_dict[typ] = col
    type_dict[col] = typ

options = list(color_dict.keys())
columns = ["Type", "X0", "Y0", "X1", "Y1"]


################################################################################## Functions ##################################################################################
# def debug_print(*args):
#     if DEBUG:
#         print(*args)


def coord_to_tab_column(coord):
    return coord.upper()


def time_passed(start=0):
    return round(time.mktime(time.localtime())) - start


def format_float(f):
    return "%.2f" % (float(f),)


def shape_to_table_row(sh):
    return {
        "Type": type_dict[sh["line"]["color"]],
        "X0": format_float(sh["x0"]),
        "Y0": format_float(sh["y0"]),
        "X1": format_float(sh["x1"]),
        "Y1": format_float(sh["y1"]),
    }


def default_table_row():
    return {
        "Type": DEFAULT_ATYPE,
        "X0": format_float(10),
        "Y0": format_float(10),
        "X1": format_float(20),
        "Y1": format_float(20),
    }


def table_row_to_shape(tr):
    return {
        # "editable": True,
        # "xref": "x",
        # "yref": "y",
        # "layer": "above",
        # "opacity": 1,
        "line": {"color": color_dict[tr["Type"]], "width": 4, "dash": "solid"},
        # "fillcolor": "rgba(0, 0, 0, 0)",
        # "fillrule": "evenodd",
        # "type": "rect",
        "x0": tr["X0"],
        "y0": tr["Y0"],
        "x1": tr["X1"],
        "y1": tr["Y1"],
    }


def shape_cmp(s0, s1):
    """ Compare two shapes """
    return (
        (s0["x0"] == s1["x0"])
        and (s0["x1"] == s1["x1"])
        and (s0["y0"] == s1["y0"])
        and (s0["y1"] == s1["y1"])
        and (s0["line"]["color"] == s1["line"]["color"])
    )


def shape_in(se):
    """ check if a shape is in list (done this way to use custom compare) """
    return lambda s: any(shape_cmp(s, s_) for s_ in se)


def index_of_shape(shapes, shape):
    for i, shapes_item in enumerate(shapes):
        if shape_cmp(shapes_item, shape):
            return i
    raise ValueError  # not found


def annotations_table_shape_resize(annotations_table_data, fig_data):
    """
    Extract the shape that was resized (its index) and store the resized
    coordinates.
    """
    # debug_print("fig_data", fig_data)
    # debug_print("table_data", annotations_table_data)
    for key, val in fig_data.items():
        shape_nb, coord = key.split(".")
        # shape_nb is for example 'shapes[2].x0': this extracts the number
        shape_nb = shape_nb.split(".")[0].split("[")[-1].split("]")[0]
        # this should correspond to the same row in the data table
        # we have to format the float here because this is exactly the entry in
        # the table
        annotations_table_data[int(shape_nb)][
            coord_to_tab_column(coord)
        ] = format_float(fig_data[key])
        # (no need to compute a time stamp, that is done for any change in the
        # table values, so will be done later)
    return annotations_table_data


def shape_data_remove_timestamp(shape):
    """
    go.Figure complains if we include the 'timestamp' key when updating the
    figure
    """
    new_shape = dict()
    for k in shape.keys() - set(["timestamp"]):
        new_shape[k] = shape[k]
    return new_shape

################################################################################## DASH ##################################################################################
external_stylesheets = ['assets/image_annotation_style.css',
                        "https://codepen.io/chriddyp/pen/bWLwgP.css",
                        "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"]

app = dash.Dash(external_stylesheets=external_stylesheets)

server = app.server

fig = px.imshow(io.imread(filelist[0]), binary_backend="jpg")
fig.update_layout(
    newshape_line_color=color_dict[DEFAULT_ATYPE],
    margin=dict(l=0, r=0, b=0, t=0, pad=4),
    dragmode="drawrect",
    hovermode=False
)

# Buttons
# button_gh = dbc.Button(
#     "Learn more",
#     id="howto-open",
#     outline=True,
#     color="secondary",
#     # Turn off lowercase transformation for class .button in stylesheet
#     style={"textTransform": "none"},
# )

# button_howto = dbc.Button(
#     "View Code on github",
#     outline=True,
#     color="primary",
#     href="https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-image-annotation",
#     id="gh-link",
#     style={"text-transform": "none"},
# )

# Modal
# modal_overlay = dbc.Modal(
#     [
#         dbc.ModalBody(html.Div([dcc.Markdown(howto, id="howto-md")])),
#         dbc.ModalFooter(dbc.Button("Close", id="howto-close", className="howto-bn",)),
#     ],
#     id="modal",
#     size="lg",
#     style={"font-size": "small"},
# )

# Cards
image_annotation_card = dbc.Card(
    id="imagebox",
    children=[
        dbc.CardHeader(
            [
                html.H2("Annotation area ,, Image No 1")
            ],
            id='image_counter'),

        dbc.CardBody(
            [
                dcc.Graph(
                    id="graph",
                    figure=fig,
                    config={"modeBarButtonsToAdd": ["drawrect", "eraseshape"]},
                )
            ]
        ),
        dbc.CardFooter(
            [
                dcc.Markdown(
                    "**Choose a different image to annotate**:"
                ),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Previous image", id="previous", outline=True),
                        dbc.Button("Next image", id="next", outline=True),
                    ],
                    size="lg",
                    style={"width": "100%"},
                ),
            ]
        ),
    ],
)

annotated_data_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Annotated data")),
        dbc.CardBody(
            [
                dbc.Row(dbc.Col(html.H3("Coordinates of annotations"))),
                dbc.Row(
                    dbc.Col(
                        [
                            dash_table.DataTable(
                                id="annotations-table",
                                columns=[
                                    dict(
                                        name=n,
                                        id=n,
                                        presentation=(
                                            "dropdown" if n == "Type" else "input"
                                        ),
                                    )
                                    for n in columns
                                ],
                                editable=True,
                                style_data={"height": 40},
                                style_cell={
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                    "maxWidth": 0,
                                },
                                dropdown={
                                    "Type": {
                                        "options": [
                                            {"label": o, "value": o}
                                            for o in annotation_types
                                        ],
                                        "clearable": False,
                                    }
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": "Type"}, "textAlign": "left",}
                                ],
                                fill_width=True,
                            ),
                            dcc.Store(id="graph-copy", data=fig),
                            dcc.Store(
                                id="annotations-store",
                                data=dict(
                                    **{
                                        filename: {"shapes": []}
                                        for filename in filelist
                                    },
                                    **{"starttime": time_passed()}
                                ),
                            ),
                            dcc.Store(
                                id="image_files",
                                data={"files": filelist, "current": 0},
                            ),
                        ],
                    ),
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            html.H3("Create new annotation for"),
                            dcc.Dropdown(
                                id="annotation-type-dropdown",
                                options=[
                                    {"label": t, "value": t} for t in annotation_types
                                ],
                                value=DEFAULT_ATYPE,
                                clearable=False,
                            ),
                        ],
                        align="center",
                    )
                ),
            ]
        ),
        dbc.CardFooter(
            [
                # html.Div(
                #     [
                #         # We use this pattern because we want to be able to download the
                #         # annotations by clicking on a button
                #         html.A(
                #             id="download",
                #             download="annotations.json",
                #             # make invisble, we just want it to click on it
                #             style={"display": "none"},
                #         ),
                #         dbc.Button(
                #             "Download annotations", id="download-button", outline=True,
                #         ),
                #         html.Div(id="dummy", style={"display": "none"}),
                #         dbc.Tooltip(
                #             "You can download the annotated data in a .json format by clicking this button",
                #             target="download-button",
                #         ),
                #     ],
                # ),

                html.Div(
                    [
                        dbc.Button(
                            "Download Annotations Into TXT", id="download-txt", outline=True,
                        ),
                        html.Div(id="download-msg", style={"text-color": "green"})
                    ],
                )
            ]
        ),
    ],
)

# Navbar
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.A(
                            html.Img(
                                src=app.get_asset_url("dash-logo-new.png"),
                                height="30px",
                            ),
                            href="https://plot.ly",
                        )
                    ),
                    dbc.Col(dbc.NavbarBrand("Image Annotation App")),
                ],
                align="center",
            )
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    className="mb-5",
)

app.layout = html.Div(
    [
        navbar,
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(image_annotation_card, md=7),
                        dbc.Col(annotated_data_card, md=5),
                    ],
                ),
            ],
            fluid=True,
        ),
    ]
)

################################################################################## CALLBACKS ##################################################################################
def format_annotation(a, image_path):
    class_num = annotations_class[a['Type']]
    x0, x1, y0, y1 = float(a['X0']), float(a['X1']), float(a['Y0']), float(a['Y1'])  
    xmin, xmax = [x0, x1] if x0 < x1 else [x1, x0] 
    ymin, ymax = [y0, y1] if y0 < y1 else [y1, y0] 

    # Transform the bbox co-ordinates as per the format required by YOLO 
    b_center_x = (xmin + xmax) / 2 
    b_center_y = (ymin + ymax) / 2
    b_width    = (xmax - xmin)
    b_height   = (ymax - ymin)

    # Normalise the co-ordinates by the dimensions of the image
    img = cv2.imread(image_path)

    # Get the shape of the image
    image_h, image_w, _ =  img.shape
    b_center_x /= image_w 
    b_center_y /= image_h 
    b_width    /= image_w
    b_height   /= image_h 
    
    return " ".join(str(num) for num in [class_num, b_center_x, b_center_y, b_width, b_height])

@app.callback(
        [Output("download-msg", "children")],
        [Input("download-txt", "n_clicks")],
        [
            State("annotations-table", "data"),
            State("image_files", "data")
        ]
)
def save_annotation(n_clicks, annotations, images):
    if annotations:
        current_image_index = images['current']
        current_image = images['files'][current_image_index]
        ann_file_name = current_image.split('/')[1] + '.txt'

        # Check if the directory already exists
        if not os.path.exists(ANN_DIR_PATH):
            # If it doesn't exist, create the directory
            os.makedirs(ANN_DIR_PATH)

        with open(ANN_DIR_PATH + ann_file_name, 'w') as file:
            for a in annotations:
                file.write(format_annotation(a, current_image))
                file.write("\n")

        annotataed_files.append(ann_file_name)
        return [html.H4("DONE " + str(n_clicks), style={'color': 'green'})]
    
    return [""]


@app.callback(
        [
            Output('image_counter', "children")
        ],

        [
            Input("previous", "n_clicks"),
            Input("next", "n_clicks"),
        ],
        [
            State("image_files", "data")
        ]
)
def update_image_counter(prev_n_clicks, next_n_clicks, images):
    check = ""
    if next_n_clicks and prev_n_clicks:
        counter = next_n_clicks - prev_n_clicks + 1
    
    elif next_n_clicks:
        counter = next_n_clicks + 1
    
    else:
        counter = 1

    # current_image_index = images['current']-1
    current_image_index = abs(counter)%(len(images['files'])+1)-1
    current_image = images['files'][current_image_index]
    ann_file_name = current_image.split('/')[1] + '.txt'

    if ann_file_name in annotataed_files:
        check = "Annotated"

    return [html.H2("Annotation area ,, Image No " + str(current_image_index+2) + "   " + check)]
    



def get_next_ann_class(ann_type):
    indx = annotation_types.index(ann_type)
    next_ann = annotation_types[(indx+1)%len(annotation_types)]
    return next_ann

@app.callback(
    [
        Output("annotations-table", "data"), 
        Output("image_files", "data"),
        Output("annotation-type-dropdown", "value")
    ],
    [
        Input("previous", "n_clicks"),
        Input("next", "n_clicks"),
        Input("graph", "relayoutData"),
    ],
    [
        State("annotations-table", "data"),
        State("image_files", "data"),
        State("annotations-store", "data"),
        State("annotation-type-dropdown", "value"),
    ],
)
def modify_table_entries(
    previous_n_clicks,
    next_n_clicks,
    graph_relayoutData,
    annotations_table_data,
    image_files_data,
    annotations_store_data,
    annotation_type,
):
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if cbcontext == "graph.relayoutData":
        # debug_print("graph_relayoutData:", graph_relayoutData)
        # debug_print("annotations_table_data before:", annotations_table_data)
        if "shapes" in graph_relayoutData.keys():
            # this means all the shapes have been passed to this function via
            # graph_relayoutData, so we store them
            annotations_table_data = [
                shape_to_table_row(sh) for sh in graph_relayoutData["shapes"]
            ]
        elif re.match("shapes\[[0-9]+\].x0", list(graph_relayoutData.keys())[0]):
            # this means a shape was updated (e.g., by clicking and dragging its
            # vertices), so we just update the specific shape
            annotations_table_data = annotations_table_shape_resize(
                annotations_table_data, graph_relayoutData
            )
        if annotations_table_data is None:
            return dash.no_update
        else:
            # debug_print("annotations_table_data after:", annotations_table_data)
            next_ann = get_next_ann_class(annotation_type)
            return (annotations_table_data, image_files_data, next_ann)
        
    image_index_change = 0
    if cbcontext == "previous.n_clicks":
        image_index_change = -1
    if cbcontext == "next.n_clicks":
        image_index_change = 1
    image_files_data["current"] += image_index_change
    image_files_data["current"] %= len(image_files_data["files"])
    if image_index_change != 0:
        # image changed, update annotations_table_data with new data
        annotations_table_data = []
        filename = image_files_data["files"][image_files_data["current"]]
        # debug_print(annotations_store_data[filename])
        for sh in annotations_store_data[filename]["shapes"]:
            annotations_table_data.append(shape_to_table_row(sh)) 

        # next_ann = get_next_ann_class(annotation_type)       
        return (annotations_table_data, image_files_data, DEFAULT_ATYPE)
    else:
        return dash.no_update


@app.callback(
    [Output("graph", "figure"), Output("annotations-store", "data"),],
    [Input("annotations-table", "data"), Input("annotation-type-dropdown", "value")],
    [State("image_files", "data"), State("annotations-store", "data")],
)
def send_figure_to_graph(
    annotations_table_data, annotation_type, image_files_data, annotations_store
):
    if annotations_table_data is not None:
        filename = image_files_data["files"][image_files_data["current"]]
        # convert table rows to those understood by fig.update_layout
        fig_shapes = [table_row_to_shape(sh) for sh in annotations_table_data]
        # debug_print("fig_shapes:", fig_shapes)
        # debug_print(
        #     "annotations_store[%s]['shapes']:" % (filename,),
        #     annotations_store[filename]["shapes"],
        # )
        # find the shapes that are new
        new_shapes_i = []
        old_shapes_i = []
        for i, sh in enumerate(fig_shapes):
            if not shape_in(annotations_store[filename]["shapes"])(sh):
                new_shapes_i.append(i)
            else:
                old_shapes_i.append(i)
        # add timestamps to the new shapes
        for i in new_shapes_i:
            fig_shapes[i]["timestamp"] = time_passed(annotations_store["starttime"])
        # find the old shapes and look up their timestamps
        for i in old_shapes_i:
            old_shape_i = index_of_shape(
                annotations_store[filename]["shapes"], fig_shapes[i]
            )
            fig_shapes[i]["timestamp"] = annotations_store[filename]["shapes"][
                old_shape_i
            ]["timestamp"]
        shapes = fig_shapes
        # debug_print("shapes:", shapes)
        fig = px.imshow(io.imread(filename), binary_backend="jpg")
        fig.update_layout(
            shapes=[shape_data_remove_timestamp(sh) for sh in shapes],
            # reduce space between image and graph edges
            newshape_line_color=color_dict[annotation_type],
            margin=dict(l=0, r=0, b=0, t=0, pad=4),
            dragmode="drawrect",
            hovermode=False
        )
        annotations_store[filename]["shapes"] = shapes
        return (fig, annotations_store)
    return dash.no_update


# @app.callback(
#     Output("modal", "is_open"),
#     [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
#     [State("modal", "is_open")],
# )
# def toggle_modal(n1, n2, is_open):
#     if n1 or n2:
#         return not is_open
#     return is_open


# set the download url to the contents of the annotations-store (so they can be
# downloaded from the browser's memory)
app.clientside_callback(
    """
function(the_store_data) {
    let s = JSON.stringify(the_store_data);
    let b = new Blob([s],{type: 'text/plain'});
    let url = URL.createObjectURL(b);
    return url;
}
""",
    Output("download", "href"),
    [Input("annotations-store", "data")],
)

# click on download link via button
app.clientside_callback(
    """
function(download_button_n_clicks)
{
    let download_a=document.getElementById("download");
    download_a.click();
    return '';
}
""",
    Output("dummy", "children"),
    [Input("download-button", "n_clicks")],
)


# TODO comment the dbc link
# we use a callback to toggle the collapse on small screens
# @app.callback(
#     Output("navbar-collapse", "is_open"),
#     [Input("navbar-toggler", "n_clicks")],
#     [State("navbar-collapse", "is_open")],
# )
# def toggle_navbar_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open


app.run_server()