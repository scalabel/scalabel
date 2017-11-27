(function () {
    // Global variables
    var imageCanvas = '#image_canvas';
    var ctx = $(imageCanvas)[0].getContext('2d');


    var imageCanvasWidth = $(imageCanvas).css('width');
    var imageCanvasHeight = $(imageCanvas).css('height');

    var tag;

    // BBoxLabeling Class
    this.BBoxLabeling = (function () {

        function BBoxLabeling(options) {

            this.options = options;
            // Initialize main canvas
            this.image_canvas = $('#image_canvas');
            //Load the image
            this.image_canvas.css({
                "background-image": "url('" + this.options.url + "')"
            });
            return this.eventController();

        }

        BBoxLabeling.prototype.replay = function () {
            ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
            var labels = image_list[current_index].labels;
            tag = $("select#category_select").val();
            if (labels) {
                tag = labels[0]
            }
            $("#weather_select").prop("selectedIndex",
                assignment.category.indexOf(tag));
            this.drawCaption();
        };

        BBoxLabeling.prototype.updateImage = function (url) {
            $("#image_id").val(current_index + 1);
            this.options.url = url;
            var source_image = new Image();
            source_image.src = url;
            this.image_canvas.css({
                "background-image": "url('" + url + "')"
            });
            this.output_labels = [];
            this.output_tags = [];
        };

        BBoxLabeling.prototype.submitLabels = function () {
            this.output_tags = [tag]
        };

        BBoxLabeling.prototype.eventController = function () {
            bboxLabeling = this;
            $("#category_select").change(function () {
                var cat_idx = $(this)[0].selectedIndex;
                tag = assignment.category[cat_idx];
                bboxLabeling.drawCaption();
            });

            $(document).on('keydown', function (e) {
                switch(e.keyCode) {
                    // first row of keyboard "qwertyu"
                    case 81:
                        tag = assignment.category[0];
                        break;
                    case 87:
                        tag = assignment.category[1];
                        break;
                    case 69:
                        tag = assignment.category[2];
                        break;
                    case 82:
                        tag = assignment.category[3];
                        break;
                    case 84:
                        tag = assignment.category[4];
                        break;
                    case 89:
                        tag = assignment.category[5];
                        break;
                    case 85:
                        tag = assignment.category[6];
                        break;
                    // go to previous
                    case 37:
                        goToImage(current_index - 1);
                        break;
                    // go to next
                    case 39:
                        goToImage(current_index + 1);
                        break;
                }
                bboxLabeling.drawCaption();

            });


        };

        BBoxLabeling.prototype.drawCaption = function () {
            ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
            if (typeof tag !== "undefined") {
                ctx.font = "30px Arial";
                ctx.fillStyle = "#829356";
                ctx.fillRect(10, 10, 230, 45);
                ctx.fillStyle = "White";
                ctx.fillText(tag, 15, 40);
                $("#category_select").prop("selectedIndex", assignment.category.indexOf(tag));
            }
        };

        return BBoxLabeling;

    })();

}).call(this);
