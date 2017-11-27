(function () {
    // Global variables
    var rect_dict = {};
    var imageCanvas = '#image_canvas';
    var pickupCanvas = '#pickup_canvas';
    var ctx = $(imageCanvas)[0].getContext('2d');
    var ghost_ctx = $(pickupCanvas)[0].getContext('2d');

    var main_canvas = document.getElementById("image_canvas");
    var hidden_canvas = document.getElementById("pickup_canvas");

    var offsetLeft = main_canvas.getBoundingClientRect().left;
    var offsetTop = main_canvas.getBoundingClientRect().top;
    var imageCanvasWidth = $(imageCanvas).css('width');
    var imageCanvasHeight = $(imageCanvas).css('height');
    var state = "free";
    var hide_labels = false;

    var LINE_WIDTH = 2;
    var HIDDEN_LINE_WIDTH = 4;
    var HANDLE_RADIUS = 4;
    var HIDDEN_HANDLE_RADIUS = 5;
    var TAG_WIDTH = 25;
    var TAG_HEIGHT = 14;

    var ratio;

    function CanvasResize() {

        ratio = parseFloat(window.innerWidth / (1.35 * main_canvas.width));
        if (parseFloat(window.innerHeight / (1.35 * main_canvas.height)) < ratio)
            ratio = parseFloat(window.innerHeight / (1.35 * main_canvas.height));
        ratio = parseFloat(ratio.toFixed(6));

        main_canvas.width = Math.round(main_canvas.width * ratio);
        main_canvas.height = Math.round(main_canvas.height * ratio);
        hidden_canvas.width = Math.round(hidden_canvas.width * ratio);
        hidden_canvas.height = Math.round(hidden_canvas.height * ratio);

        imageCanvasWidth = $(imageCanvas).attr('width');
        imageCanvasHeight = $(imageCanvas).attr('height');
        // Anti-aliasing
        if (window.devicePixelRatio) {
            var imageCanvasCssWidth = imageCanvasWidth;
            var imageCanvasCssHeight = imageCanvasHeight;

            $(imageCanvas).attr('width', imageCanvasWidth * window.devicePixelRatio);
            $(imageCanvas).attr('height', imageCanvasHeight * window.devicePixelRatio);
            $(imageCanvas).css('width', imageCanvasCssWidth);
            $(imageCanvas).css('height', imageCanvasCssHeight);
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }

    }

    // Global functions
    function point(x, y) {
        return {
            x: x,
            y: y
        };
    }

    function dist(p1, p2) {
        return Math.sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
    }

    var bbox_handles = [
        function (rect) { // TOP_LEFT: 0
            return point(rect.x, rect.y);
        },
        function (rect) { // TOP_RIGHT: 1
            return point(rect.x + rect.w, rect.y);
        },
        function (rect) { // BOTTOM_LEFT: 2
            return point(rect.x, rect.y + rect.h);
        },
        function (rect) { // BOTTOM_RIGHT: 3
            return point(rect.x + rect.w, rect.y + rect.h);
        },
        function (rect) { // TOP: 4
            return point(rect.x + rect.w / 2, rect.y);
        },
        function (rect) { // LEFT: 5
            return point(rect.x, rect.y + rect.h / 2);
        },
        function (rect) { // BOTTOM: 6
            return point(rect.x + rect.w / 2, rect.y + rect.h);
        },
        function (rect) { // RIGHT: 7
            return point(rect.x + rect.w, rect.y + rect.h / 2);
        }
    ];

    var drag_handle = [
        function (rect, mousePos) {
            rect.w = rect.w + rect.x - mousePos.x;
            rect.h = rect.h + rect.y - mousePos.y;
            rect.x = mousePos.x;
            rect.y = mousePos.y;
        },
        function (rect, mousePos) {
            rect.w = mousePos.x - rect.x;
            rect.h = rect.h + rect.y - mousePos.y;
            rect.y = mousePos.y;
        },
        function (rect, mousePos) {
            rect.w = rect.w + rect.x - mousePos.x;
            rect.x = mousePos.x;
            rect.h = mousePos.y - rect.y;
        },
        function (rect, mousePos) {
            rect.w = mousePos.x - rect.x;
            rect.h = mousePos.y - rect.y;
        },
        function (rect, mousePos) {
            rect.h = rect.h + rect.y - mousePos.y;
            rect.y = mousePos.y;

        },
        function (rect, mousePos) {
            rect.w = rect.w + rect.x - mousePos.x;
            rect.x = mousePos.x;
        },
        function (rect, mousePos) {
            rect.h = mousePos.y - rect.y;

        },
        function (rect, mousePos) {
            rect.w = mousePos.x - rect.x;

        }
    ];

    // BBoxLabeling Class
    this.BBoxLabeling = (function () {

        function BBoxLabeling(options) {

            this.options = options;
            // Initialize main canvas
            this.image_canvas = $('#image_canvas');
            // this.pickup_canvas = $('#pickup_canvas');
            //Load the image
            this.image_canvas.css({
                "background-image": "url('" + this.options.url + "')",
                "cursor": "crosshair"
            });
            return this.eventController();

        }

        BBoxLabeling.prototype.replay = function () {
            this.updateImage(image_list[current_index].url);
            var labels = image_list[current_index].labels;
            ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
            ghost_ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
            if (labels) {
                for (var key in labels) {
                    if (labels.hasOwnProperty(key)) {
                        var label = labels[key];
                        var rect = new BBox(label.category, label.id,
                            [false, false, "none"]);
                        if (label.position) {
                            rect.x = parseFloat((label.position.x1 * ratio).toFixed(6));
                            rect.y = parseFloat((label.position.y1 * ratio).toFixed(6));
                            rect.w = parseFloat(((label.position.x2 - label.position.x1) * ratio).toFixed(6));
                            rect.h = parseFloat(((label.position.y2 - label.position.y1) * ratio).toFixed(6));
                        }
                        if (label.category) {
                            rect.category = label.category;
                        }
                        if (label.attribute && label.attribute.occluded) {
                            rect.occluded = label.attribute.occluded;
                        }
                        if (label.attribute && label.attribute.truncated) {
                            rect.truncated = label.attribute.truncated;
                        }
                        if (label.attribute && label.attribute.traffic_light_color) {
                            rect.traffic_light_color = label.attribute.traffic_light_color;
                        }

                        rect.id = parseInt(label.id);
                        rect_dict[rect.id] = rect;

                        rect.drawBox();
                        rect.drawHiddenBox();
                        rect.drawTag();
                    }
                }
            }
        };

        BBoxLabeling.prototype.updateImage = function (url) {
            this.options.url = url;
            var source_image = new Image();
            source_image.src = url;
            this.image_canvas.css({
                "background-image": "url('" + url + "')"
            });
            if (source_image.complete) {
                main_canvas.width = source_image.width;
                main_canvas.height = source_image.height;
                hidden_canvas.width = source_image.width;
                hidden_canvas.height = source_image.height;
                CanvasResize();
            } else {
                source_image.onload = function () {
                    main_canvas.width = source_image.width;
                    main_canvas.height = source_image.height;
                    hidden_canvas.width = source_image.width;
                    hidden_canvas.height = source_image.height;
                    CanvasResize();
                }
            }
            rect_dict = {};
        };

        BBoxLabeling.prototype.submitLabels = function () {
            this.output_labels = [];
            for (var key in rect_dict) {
                var rect = rect_dict[key];
                var output = {
                    position: {
                        x1: parseFloat((Math.min(rect.x, rect.x + rect.w) / ratio).toFixed(6)),
                        y1: parseFloat((Math.min(rect.y, rect.y + rect.h) / ratio).toFixed(6)),
                        x2: parseFloat((Math.max(rect.x, rect.x + rect.w) / ratio).toFixed(6)),
                        y2: parseFloat((Math.max(rect.y, rect.y + rect.h) / ratio).toFixed(6))
                    },
                    category: rect.category,
                    id: rect.id.toString(),
                    attribute: {
                        occluded: rect.occluded,
                        truncated: rect.truncated,
                        traffic_light_color: rect.traffic_light_color
                    }
                };
                this.output_labels.push(output);
            }
        };

        BBoxLabeling.prototype.clearAll = function () {
            rect_dict = {};
            ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
            ghost_ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
            state = "free";
        };

        BBoxLabeling.prototype.getSelectedBbox = function (mouse) {
            var pixelData = ghost_ctx.getImageData(mouse.x, mouse.y, 1, 1).data;
            var current_handle;
            var selected_bbox;
            if (pixelData[0] !== 0 && pixelData[3] === 255) {
                var rect_id = pixelData[0] - 1;
                current_handle = pixelData[1] - 1;
                selected_bbox = rect_dict[rect_id];
            } else {
                current_handle = -1;
                selected_bbox = -1;
            }
            return [current_handle, selected_bbox]
        };

        BBoxLabeling.prototype.highlight = function (bbox) {
            if (bbox !== -1) {

                ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);

                ctx.globalAlpha = 0.5;
                ctx.setLineDash([]);
                for (key in rect_dict) {
                    if (key !== bbox.id.toString()) {
                        var cur = rect_dict[key];
                        cur.drawBox();
                        cur.drawTag();
                    }
                }
                ctx.globalAlpha = 1.0;
                bbox.drawBox();
                bbox.drawHiddenBox();
                bbox.drawTag();

                for (var h = 0; h <= 7; h++) {
                    bbox.drawHandle(h);
                }
                $('#toolbox').css("background-color", "#67b168");
            }
        };

        BBoxLabeling.prototype.eventController = function () {
            var rect = -1;
            var selected_bbox = -1;
            var current_bbox = -1;
            var current_handle = -1;
            var previous_handle = -1;
            var bboxLabeling = this;

            $("#category_select").change(function () {
                if (current_bbox !== -1 && typeof(current_bbox) !== "undefined") {
                    var cat_idx = $(this)[0].selectedIndex;
                    if (assignment.category[cat_idx] === "traffic light") {
                        num_light = num_light + 1;
                    }
                    if (rect_dict[current_bbox.id].category === "traffic light") {
                        num_light = num_light - 1;
                    }
                    rect_dict[current_bbox.id].category = assignment.category[cat_idx];
                    bboxLabeling.highlight(current_bbox);
                }
            });
            $("[name='occluded-checkbox']").on('switchChange.bootstrapSwitch', function (event, state) {
                if (current_bbox !== -1 && typeof(current_bbox) !== "undefined") {
                    rect_dict[current_bbox.id].occluded = $(this).prop("checked");
                    bboxLabeling.highlight(current_bbox);
                }
            });

            $("[name='truncated-checkbox']").on('switchChange.bootstrapSwitch', function (event, state) {
                if (current_bbox !== -1 && typeof(current_bbox) !== "undefined") {
                    rect_dict[current_bbox.id].truncated = $(this).prop("checked");
                    bboxLabeling.highlight(current_bbox);
                }
            });

            $("#radios :input").change(function () {
                if (current_bbox !== -1 && typeof(current_bbox) !== "undefined") {
                    rect_dict[current_bbox.id].traffic_light_color =
                        $("input[type='radio']:checked").attr("id");
                    bboxLabeling.highlight(current_bbox);
                }
            });

            $(document).on('keydown', function (e) {
                // keyboard shortcut for delete
                if (e.which === 8 || e.which === 46) {
                    if (current_bbox !== -1 && typeof(current_bbox) !== "undefined") {
                        current_bbox.removeBox();
                    }
                    state = "free";
                    $('#toolbox').css("background-color", "#DCDCDC");
                    current_bbox = -1;
                    rect = -1;
                }
                // keyboard shortcut for hiding labels
                if (e.keyCode === 72) {
                    if (!hide_labels) {
                        hide_labels = true;
                    } else {
                        hide_labels = false;
                    }
                    bboxLabeling.image_canvas.css("cursor", "crosshair");
                    ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
                    ctx.setLineDash([]);
                    for (var key in rect_dict) {
                        var cur = rect_dict[key];
                        cur.drawBox();
                        cur.drawHiddenBox();
                        cur.drawTag();
                    }
                }
                // "A" key used for checking occluded box
                if (e.keyCode === 65) {
                    if (current_bbox !== -1 && typeof(current_bbox) !== "undefined") {
                        $("[name='occluded-checkbox']").trigger('click');
                        rect_dict[current_bbox.id].occluded = $("[name='occluded-checkbox']").prop("checked");
                        bboxLabeling.highlight(current_bbox);
                    }
                }

                // "E" key used for checking truncated box
                if (e.keyCode === 83) {
                    if (current_bbox !== -1 && typeof(current_bbox) !== "undefined") {
                        $("[name='truncated-checkbox']").trigger('click');
                        rect_dict[current_bbox.id].truncated = $("[name='truncated-checkbox']").prop("checked");
                        bboxLabeling.highlight(current_bbox);
                    }
                }
            });

            $("#remove_btn").click(function () {
                if (current_bbox !== -1 && typeof(current_bbox) !== "undefined") {
                    current_bbox.removeBox();
                }
                state = "free";
                $('#toolbox').css("background-color", "#DCDCDC");
                current_bbox = -1;
                rect = -1;
            });

            $(document).on('mousemove', '#image_canvas', function (e) {

                // Full-canvas crosshair mouse cursor
                var cH = $('#crosshair-h'),
                    cV = $('#crosshair-v');
                $(".hair").show();
                var x = e.clientX;
                var y = e.clientY;
                cH.css('top', Math.max(y, main_canvas.getBoundingClientRect().top));
                cH.css('left', main_canvas.getBoundingClientRect().left);
                cH.css('width', imageCanvasWidth);

                cV.css('right', main_canvas.getBoundingClientRect().right);
                cV.css('left', Math.max(x, main_canvas.getBoundingClientRect().left));
                cV.css('height', imageCanvasHeight);

                offsetLeft = main_canvas.getBoundingClientRect().left;
                offsetTop = main_canvas.getBoundingClientRect().top;

                if (state === "hover_resize" || state === "select_resize") {
                    if (rect !== -1 && typeof(rect) !== "undefined") {
                        var mousePos = point(e.clientX - offsetLeft, e.clientY - offsetTop);
                        if (current_handle >= 0 && current_handle <= 7) {
                            drag_handle[current_handle](rect, mousePos);
                        }
                        ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);

                        ctx.globalAlpha = 0.5;
                        ctx.setLineDash([]);
                        for (key in rect_dict) {
                            if (key !== rect.id.toString()) {
                                var cur = rect_dict[key];
                                cur.drawBox();
                                cur.drawTag();
                            }
                        }
                        ctx.globalAlpha = 1.0;
                        ctx.setLineDash([3]);
                        rect.drawBox();
                        if (current_handle >= 0 && current_handle <= 7) {
                            ctx.setLineDash([]);
                            rect.drawHandle(current_handle);
                        }
                    }

                } else if (state === "draw") {
                    rect.update(e.clientX, e.clientY);

                } else if (state === "select") {
                    bboxLabeling.highlight(current_bbox);

                } else {
                    // hover on
                    mousePos = point(e.clientX - offsetLeft, e.clientY - offsetTop);
                    previous_handle = current_handle;

                    var return_value;
                    return_value = bboxLabeling.getSelectedBbox(mousePos);
                    current_handle = return_value[0];
                    selected_bbox = return_value[1];

                    if (selected_bbox !== -1 && typeof(selected_bbox) !== "undefined") {
                        if (current_handle >= 0 && current_handle <= 7) {
                            var handlePos = bbox_handles[current_handle](selected_bbox);
                            if (dist(mousePos, handlePos) < HIDDEN_HANDLE_RADIUS - 2) {
                                selected_bbox.drawHandle(current_handle);
                            }

                        } else if (current_handle === 8) {
                            bboxLabeling.image_canvas.css("cursor", "pointer");
                        }
                    }

                    if (current_handle !== previous_handle) {
                        bboxLabeling.image_canvas.css("cursor", "crosshair");
                        ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
                        ctx.setLineDash([]);
                        for (var key in rect_dict) {
                            var cur = rect_dict[key];
                            cur.drawBox();
                            cur.drawHiddenBox();
                            cur.drawTag();
                        }
                    }

                }


            });

            $(document).on('mousedown', '#image_canvas', function (e) {

                offsetLeft = main_canvas.getBoundingClientRect().left;
                offsetTop = main_canvas.getBoundingClientRect().top;
                var mousePos = point(e.clientX - offsetLeft, e.clientY - offsetTop);
                var return_value;
                return_value = bboxLabeling.getSelectedBbox(mousePos);
                current_handle = return_value[0];
                selected_bbox = return_value[1];
                if (current_handle >= 0 && current_handle <= 7) {
                    rect = selected_bbox;
                    addEvent("resize bbox", rect.id, {
                        x1: parseFloat((Math.min(rect.x, rect.x + rect.w) / ratio).toFixed(6)),
                        y1: parseFloat((Math.min(rect.y, rect.y + rect.h) / ratio).toFixed(6)),
                        x2: parseFloat((Math.max(rect.x, rect.x + rect.w) / ratio).toFixed(6)),
                        y2: parseFloat((Math.max(rect.y, rect.y + rect.h) / ratio).toFixed(6))
                    });
                    if (state === "select") {
                        state = "select_resize";
                    } else {
                        state = "hover_resize";
                    }

                } else if (current_handle === 8) {
                    current_bbox = selected_bbox;
                    rect = selected_bbox;
                    state = "select";
                    bboxLabeling.highlight(current_bbox);
                    $("#category_select").prop("selectedIndex",
                        assignment.category.indexOf(rect_dict[current_bbox.id].category));
                    if (typeof rect_dict[current_bbox.id].occluded !== "undefined" &&
                        typeof rect_dict[current_bbox.id].truncated !== "undefined") {
                        if ($("[name='occluded-checkbox']").prop("checked") !==
                            rect_dict[current_bbox.id].occluded) {
                            $("[name='occluded-checkbox']").trigger('click');
                        }
                        if ($("[name='truncated-checkbox']").prop("checked") !==
                            rect_dict[current_bbox.id].truncated) {
                            $("[name='truncated-checkbox']").trigger('click');
                        }
                    }
                    if (typeof rect_dict[current_bbox.id].traffic_light_color !== "undefined") {
                        $("input:radio[id='" + rect_dict[current_bbox.id].traffic_light_color + "']").trigger('click');
                    }

                } else {
                    // Unselect
                    if (state === "select") {
                        state = "free";
                        current_bbox = -1;
                        rect = -1;
                        bboxLabeling.image_canvas.css("cursor", "crosshair");
                        $('#toolbox').css("background-color", "#DCDCDC");
                    }
                    // Draw a new bbox
                    var cat_idx = document.getElementById("category_select").selectedIndex;
                    var cat = assignment.category[cat_idx];
                    var occluded = $("[name='occluded-checkbox']").prop("checked");
                    var truncated = $("[name='truncated-checkbox']").prop("checked");
                    var color = $("input[type='radio']:checked").attr("id");
                    rect = new BBox(cat, Object.keys(rect_dict).length, [occluded, truncated, color]);
                    rect.start(e.clientX, e.clientY);
                    state = "draw";
                }
            });

            $(document).on('mouseup', '#image_canvas', function () {
                rect.finish();
                if (Math.abs(rect.w) <= 7 && Math.abs(rect.h) <= 7) {
                    rect.removeBox();
                    state = "free";
                    $('#toolbox').css("background-color", "#DCDCDC");
                } else {
                    current_bbox = rect;
                    state = "select";
                    bboxLabeling.highlight(current_bbox);
                    $("#category_select").prop("selectedIndex",
                        assignment.category.indexOf(rect_dict[current_bbox.id].category));

                    if (typeof rect_dict[current_bbox.id].occluded !== "undefined" &&
                        typeof rect_dict[current_bbox.id].truncated !== "undefined") {
                        if ($("[name='occluded-checkbox']").prop("checked") !==
                            rect_dict[current_bbox.id].occluded) {
                            $("[name='occluded-checkbox']").trigger('click');
                        }
                        if ($("[name='truncated-checkbox']").prop("checked") !==
                            rect_dict[current_bbox.id].truncated) {
                            $("[name='truncated-checkbox']").trigger('click');
                        }
                        if (typeof rect_dict[current_bbox.id].traffic_light_color !== "undefined") {
                            $("input:radio[id='" + rect_dict[current_bbox.id].traffic_light_color + "']").trigger('click');
                        }

                    }
                }


            });
        };

        return BBoxLabeling;

    })();

    // BBox Class
    var BBox;
    BBox = (function () {
        function BBox(category, id, attribute) {
            this.x = 0;
            this.y = 0;
            this.w = 0;
            this.h = 0;
            this.category = category;
            this.occluded = attribute[0];
            this.truncated = attribute[1];
            this.traffic_light_color = attribute[2];
            this.id = id;
        }

        BBox.prototype.start = function (pageX, pageY) {
            this.x = pageX - main_canvas.getBoundingClientRect().left;
            this.y = pageY - main_canvas.getBoundingClientRect().top;
            this.w = 0;
            this.h = 0;
            addEvent("draw bbox", this.id, {
                x1: parseFloat((Math.min(this.x, this.x + this.w) / ratio).toFixed(6)),
                y1: parseFloat((Math.min(this.y, this.y + this.h) / ratio).toFixed(6)),
                x2: parseFloat((Math.max(this.x, this.x + this.w) / ratio).toFixed(6)),
                y2: parseFloat((Math.max(this.y, this.y + this.h) / ratio).toFixed(6))
            });
        };

        BBox.prototype.update = function (pageX, pageY) {
            this.w = (pageX - main_canvas.getBoundingClientRect().left) - this.x;
            this.h = (pageY - main_canvas.getBoundingClientRect().top) - this.y;

            ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);

            ctx.globalAlpha = 0.5;
            ctx.setLineDash([]);
            for (var key in rect_dict) {
                var cur = rect_dict[key];
                cur.drawBox();
                cur.drawTag();
            }
            ctx.globalAlpha = 1.0;
            ctx.setLineDash([3]);
            this.drawBox();
        };

        BBox.prototype.finish = function () {
            addEvent("finish bbox", this.id, {
                x1: parseFloat((Math.min(this.x, this.x + this.w) / ratio).toFixed(6)),
                y1: parseFloat((Math.min(this.y, this.y + this.h) / ratio).toFixed(6)),
                x2: parseFloat((Math.max(this.x, this.x + this.w) / ratio).toFixed(6)),
                y2: parseFloat((Math.max(this.y, this.y + this.h) / ratio).toFixed(6))
            });
            ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
            ghost_ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);

            ctx.globalAlpha = 0.5;
            ctx.setLineDash([]);
            for (var key in rect_dict) {
                var cur = rect_dict[key];
                cur.drawBox();
                cur.drawHiddenBox();
                cur.drawTag();
            }
            ctx.globalAlpha = 1.0;
            this.drawBox();
            this.drawHiddenBox();
            this.drawTag();

            if (this.id === Object.keys(rect_dict).length) {
                if (this.category === "traffic light") {
                    num_light = num_light + 1;
                }
                num_bbox = num_bbox + 1;
            }
            $("#bbox_count").text(num_bbox);
            $("#light_count").text(num_light);

            rect_dict[this.id] = this;
        };

        BBox.prototype.drawTag = function () {
            if (!hide_labels) {
                if (this.category && Math.abs(this.w) > 7 && Math.abs(this.h) > 7) {
                    var x1 = Math.min(this.x, this.x + this.w);
                    var y1 = Math.min(this.y, this.y + this.h);
                    ctx.font = "11px Verdana";
                    var tag_width = TAG_WIDTH;
                    var words = this.category.split(" ");
                    var abbr = words[words.length - 1].substring(0, 3);
                    if (this.occluded) {
                        abbr += "," + "o";
                        tag_width += 9;
                    }
                    if (this.truncated) {
                        abbr += "," + "t";
                        tag_width += 9;
                    }
                    if (this.traffic_light_color && this.traffic_light_color !== "none") {
                        abbr += "," + this.traffic_light_color.substring(0, 1);
                        tag_width += 9;
                    }
                    ctx.fillStyle = this.colors(this.id);
                    ctx.fillRect(x1 - 1, y1 - TAG_HEIGHT, tag_width, TAG_HEIGHT);
                    ctx.fillStyle = "rgb(0, 0, 0)";
                    ctx.fillText(abbr, x1 + 1, y1 - 3);
                }
            }
        };

        BBox.prototype.drawBox = function () {
            if (Math.abs(this.w) <= 7 && Math.abs(this.h) <= 7) {
                ctx.strokeStyle = "rgb(169, 169, 169)";
            } else {
                ctx.strokeStyle = this.colors(this.id);
            }
            ctx.lineWidth = LINE_WIDTH;
            ctx.strokeRect(this.x, this.y, this.w, this.h);
        };

        BBox.prototype.drawHiddenBox = function () {
            // draw hidden box frame
            ghost_ctx.lineWidth = HIDDEN_LINE_WIDTH;
            ghost_ctx.strokeStyle = this.hidden_colors(this.id, 8);
            ghost_ctx.strokeRect(this.x, this.y, this.w, this.h);

            // draw hidden tag
            if (!hide_labels) {
                var x1 = Math.min(this.x, this.x + this.w);
                var y1 = Math.min(this.y, this.y + this.h);
                ghost_ctx.fillStyle = this.hidden_colors(this.id, 8);
                ghost_ctx.fillRect(x1 - 1, y1 - TAG_HEIGHT, TAG_WIDTH, TAG_HEIGHT);

            }
            // draws eight hidden handles
            for (var i = 0; i < 8; i++) {
                this.drawHiddenHandle(i);
            }
        };

        BBox.prototype.drawHandle = function (index) {
            var handlesSize = HANDLE_RADIUS;
            var posHandle = bbox_handles[index](this);
            ctx.beginPath();
            ctx.arc(posHandle.x, posHandle.y, handlesSize, 0, 2 * Math.PI);
            ctx.fillStyle = this.colors(this.id);
            ctx.fill();

            ctx.lineWidth = 1;
            ctx.strokeStyle = "white";
            ctx.stroke();
        };

        BBox.prototype.drawHiddenHandle = function (index) {
            var handlesSize = HIDDEN_HANDLE_RADIUS;
            var posHandle = bbox_handles[index](this);
            ghost_ctx.beginPath();
            ghost_ctx.arc(posHandle.x, posHandle.y, handlesSize, 0, 2 * Math.PI);
            ghost_ctx.fillStyle = this.hidden_colors(this.id, index);
            ghost_ctx.fill();
        };

        BBox.prototype.colors = function (id) {
            var tableau_colors = [
                "rgb(31, 119, 180)",
                "rgb(174, 199, 232)",
                "rgb(255, 127, 14)",
                "rgb(255, 187, 120)",
                "rgb(44, 160, 44)",
                "rgb(152, 223, 138)",
                "rgb(214, 39, 40)",
                "rgb(255, 152, 150)",
                "rgb(148, 103, 189)",
                "rgb(197, 176, 213)",
                "rgb(140, 86, 75)",
                "rgb(196, 156, 148)",
                "rgb(227, 119, 194)",
                "rgb(247, 182, 210)",
                "rgb(127, 127, 127)",
                "rgb(199, 199, 199)",
                "rgb(188, 189, 34)",
                "rgb(219, 219, 141)",
                "rgb(23, 190, 207)",
                "rgb(158, 218, 229)"];
            return tableau_colors[id % 20]
        };

        BBox.prototype.hidden_colors = function (id, handle_index) {
            return "rgb(" + (id + 1) + "," + (handle_index + 1) + ",0)";
        };

        BBox.prototype.removeBox = function () {
            addEvent("remove bbox", this.id, {
                x1: parseFloat((Math.min(this.x, this.x + this.w) / ratio).toFixed(6)),
                y1: parseFloat((Math.min(this.y, this.y + this.h) / ratio).toFixed(6)),
                x2: parseFloat((Math.max(this.x, this.x + this.w) / ratio).toFixed(6)),
                y2: parseFloat((Math.max(this.y, this.y + this.h) / ratio).toFixed(6))
            });
            ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
            ghost_ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
            var temp_dict = rect_dict;
            rect_dict = {};
            var i = 0;
            for (var key in temp_dict) {
                var temp = temp_dict[key];
                if (key !== this.id.toString()) {
                    temp.id = i;
                    rect_dict[i] = temp;
                    temp.drawBox();
                    temp.drawHiddenBox();
                    temp.drawTag();
                }
                i++;
            }
            if (this.category === "traffic light") {
                num_light = num_light - 1;
            }
            num_bbox = num_bbox - 1;
            $("#bbox_count").text(num_bbox);
            $("#light_count").text(num_light);
        };

        return BBox;

    })();


}).call(this);
