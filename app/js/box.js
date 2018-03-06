/* global addEvent assignment:true imageList:true
currentIndex:true mousePos:true SatImage SatLabel
*/


let rectDict = {};
let imageCanvas = '#image_canvas';
let pickupCanvas = '#pickup_canvas';
let ctx = $(imageCanvas)[0].getContext('2d');
let ghostCtx = $(pickupCanvas)[0].getContext('2d');

let mainCanvas = document.getElementById('image_canvas');
let hiddenCanvas = document.getElementById('pickup_canvas');

let offsetLeft = mainCanvas.getBoundingClientRect().left;
let offsetTop = mainCanvas.getBoundingClientRect().top;
let imageCanvasWidth = $(imageCanvas).css('width');
let imageCanvasHeight = $(imageCanvas).css('height');
let state = 'free';
let hideLabels = false;
let numLight = 0;
let numBbox = 0;

let LINE_WIDTH = 2;
let HIDDEN_LINE_WIDTH = 4;
let HANDLE_RADIUS = 4;
let HIDDEN_HANDLE_RADIUS = 5;
let TAG_WIDTH = 25;
let TAG_HEIGHT = 14;

let ratio;

/**
* Summary: To be completed.
*
*/
function canvasResize() {
  ratio = parseFloat(window.innerWidth / (1.35 * mainCanvas.width));
  if (parseFloat(window.innerHeight
          / (1.35 * mainCanvas.height)) < ratio) {
    ratio = parseFloat(window.innerHeight
        / (1.35 * mainCanvas.height));
  }
  ratio = parseFloat(ratio.toFixed(6));

  mainCanvas.width = Math.round(mainCanvas.width * ratio);
  mainCanvas.height = Math.round(mainCanvas.height * ratio);
  hiddenCanvas.width = Math.round(hiddenCanvas.width * ratio);
  hiddenCanvas.height = Math.round(hiddenCanvas.height * ratio);

  imageCanvasWidth = $(imageCanvas).attr('width');
  imageCanvasHeight = $(imageCanvas).attr('height');
  // Anti-aliasing
  if (window.devicePixelRatio) {
    let imageCanvasCssWidth = imageCanvasWidth;
    let imageCanvasCssHeight = imageCanvasHeight;

    $(imageCanvas).attr('width', imageCanvasWidth
        * window.devicePixelRatio);
    $(imageCanvas).attr('height', imageCanvasHeight
        * window.devicePixelRatio);
    $(imageCanvas).css('width', imageCanvasCssWidth);
    $(imageCanvas).css('height', imageCanvasCssHeight);
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }
}

// Global functions
/**
* Summary: To be completed.
* @param {type} x: Description.
* @param {type} y: Description.
* @return {type} Description.
*/
function point(x, y) {
  return {
    x: x,
    y: y,
  };
}

/**
* Summary: To be completed.
* @param {type} p1: Description.
* @param {type} p2: Description.
* @return {type} Description.
*/
function dist(p1, p2) {
  return Math.sqrt((p2.x - p1.x) * (p2.x - p1.x)
      + (p2.y - p1.y) * (p2.y - p1.y));
}

let bboxHandles = [
  function(rect) { // TOP_LEFT: 0
    return point(rect.x, rect.y);
  },
  function(rect) { // TOP_RIGHT: 1
    return point(rect.x + rect.w, rect.y);
  },
  function(rect) { // BOTTOM_LEFT: 2
    return point(rect.x, rect.y + rect.h);
  },
  function(rect) { // BOTTOM_RIGHT: 3
    return point(rect.x + rect.w, rect.y + rect.h);
  },
  function(rect) { // TOP: 4
    return point(rect.x + rect.w / 2, rect.y);
  },
  function(rect) { // LEFT: 5
    return point(rect.x, rect.y + rect.h / 2);
  },
  function(rect) { // BOTTOM: 6
    return point(rect.x + rect.w / 2, rect.y + rect.h);
  },
  function(rect) { // RIGHT: 7
    return point(rect.x + rect.w, rect.y + rect.h / 2);
  },
];

let dragHandle = [
  function(rect, mousePos) {
    rect.w = rect.w + rect.x - mousePos.x;
    rect.h = rect.h + rect.y - mousePos.y;
    rect.x = mousePos.x;
    rect.y = mousePos.y;
  },
  function(rect, mousePos) {
    rect.w = mousePos.x - rect.x;
    rect.h = rect.h + rect.y - mousePos.y;
    rect.y = mousePos.y;
  },
  function(rect, mousePos) {
    rect.w = rect.w + rect.x - mousePos.x;
    rect.x = mousePos.x;
    rect.h = mousePos.y - rect.y;
  },
  function(rect, mousePos) {
    rect.w = mousePos.x - rect.x;
    rect.h = mousePos.y - rect.y;
  },
  function(rect, mousePos) {
    rect.h = rect.h + rect.y - mousePos.y;
    rect.y = mousePos.y;
  },
  function(rect, mousePos) {
    rect.w = rect.w + rect.x - mousePos.x;
    rect.x = mousePos.x;
  },
  function(rect, mousePos) {
    rect.h = mousePos.y - rect.y;
  },
  function(rect, mousePos) {
    rect.w = mousePos.x - rect.x;
  },
];

// BBoxLabeling Class
this.BBoxLabeling = (function() {
  /**
  * Summary: To be completed.
  * @param {type} options: Description.
  * @return {type} Description.
  */
  function BBoxLabeling(options) {
    SatImage.call(this, options);
    this.options = options;
    // Initialize main canvas
    this.image_canvas = $('#image_canvas');
    // this.pickup_canvas = $('#pickup_canvas');
    // Load the image
    this.image_canvas.css({
      'background-image': 'url(\'' + this.options.url + '\')',
      'cursor': 'crosshair',
    });
    return this.eventController();
  }

  BBoxLabeling.prototype.replay = function() {
    this.updateImage(imageList[currentIndex].url);
    let labels = imageList[currentIndex].labels;
    ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
    ghostCtx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
    if (labels) {
      for (let key in labels) {
        if (labels.hasOwnProperty(key)) {
          let label = labels[key];
          let rect = new BBox(label.category, label.id,
              [false, false, 'none']);
          if (label.position) {
            rect.x = parseFloat((label.position.x1
                * ratio).toFixed(6));
            rect.y = parseFloat((label.position.y1
                * ratio).toFixed(6));
            rect.w = parseFloat(((label.position.x2
                - label.position.x1) * ratio).toFixed(6));
            rect.h = parseFloat(((label.position.y2
                - label.position.y1) * ratio).toFixed(6));
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
          if (label.attribute
              && label.attribute.traffic_light_color) {
            rect.traffic_light_color = label.attribute.traffic_light_color;
          }

          rect.id = parseInt(label.id);
          rectDict[rect.id] = rect;

          rect.drawBox();
          rect.drawHiddenBox();
          rect.drawTag();
        }
      }
    }
  };

  BBoxLabeling.prototype.updateImage = function(url) {
    this.options.url = url;
    let sourceImage = new Image();
    sourceImage.onload = function() {
      mainCanvas.width = sourceImage.width;
      mainCanvas.height = sourceImage.height;
      hiddenCanvas.width = sourceImage.width;
      hiddenCanvas.height = sourceImage.height;
      canvasResize();
    };
    sourceImage.src = url;
    this.image_canvas.css({
      'background-image': 'url(\'' + url + '\')',
    });
    rectDict = {};
  };

  BBoxLabeling.prototype.submitLabels = function() {
    this.output_labels = [];
    for (let key in rectDict) {
      if (rectDict[key].hasOwnProperty('category')) {
        let rect = rectDict[key];
        let output = {
          position: {
            x1: parseFloat((Math.min(rect.x, rect.x + rect.w)
                / ratio).toFixed(6)),
            y1: parseFloat((Math.min(rect.y, rect.y + rect.h)
                / ratio).toFixed(6)),
            x2: parseFloat((Math.max(rect.x, rect.x + rect.w)
                / ratio).toFixed(6)),
            y2: parseFloat((Math.max(rect.y, rect.y + rect.h)
                / ratio).toFixed(6)),
          },
          category: rect.category,
          id: rect.id.toString(),
          attribute: {
            occluded: rect.occluded,
            truncated: rect.truncated,
            traffic_light_color: rect.traffic_light_color,
          },
        };
        this.output_labels.push(output);
      }
    }
  };

  BBoxLabeling.prototype.clearAll = function() {
    rectDict = {};
    ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
    ghostCtx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
    state = 'free';
  };

  BBoxLabeling.prototype.getSelectedBbox = function(mouse) {
    let pixelData = ghostCtx.getImageData(mouse.x, mouse.y, 1, 1).data;
    let currentHandle;
    let selectedBbox;
    if (pixelData[0] !== 0 && pixelData[3] === 255) {
      let rectId = pixelData[0] - 1;
      currentHandle = pixelData[1] - 1;
      selectedBbox = rectDict[rectId];
    } else {
      currentHandle = -1;
      selectedBbox = -1;
    }
    return [currentHandle, selectedBbox];
  };

  BBoxLabeling.prototype.highlight = function(bbox) {
    if (bbox !== -1) {
      ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);

      ctx.globalAlpha = 0.5;
      ctx.setLineDash([]);
      for (let key in rectDict) {
        if (key !== bbox.id.toString()) {
          let cur = rectDict[key];
          cur.drawBox();
          cur.drawTag();
        }
      }
      ctx.globalAlpha = 1.0;
      bbox.drawBox();
      bbox.drawHiddenBox();
      bbox.drawTag();

      for (let h = 0; h <= 7; h++) {
        bbox.drawHandle(h);
      }
      $('#toolbox').css('background-color', '#67b168');
    }
  };

  BBoxLabeling.prototype.eventController = function() {
    let rect = -1;
    let selectedBbox = -1;
    let currentBbox = -1;
    let currentHandle = -1;
    let previousHandle = -1;
    let bboxLabeling = this;

    $('#category_select').change(function() {
      if (currentBbox !== -1 && typeof(currentBbox) !== 'undefined') {
        let catIdx = $(this)[0].selectedIndex;
        if (assignment.category[catIdx] === 'traffic light') {
          numLight = numLight + 1;
        }
        if (rectDict[currentBbox.id].category === 'traffic light') {
          numLight = numLight - 1;
        }
        rectDict[currentBbox.id].category
            = assignment.category[catIdx];
        bboxLabeling.highlight(currentBbox);
      }
    });
    $('[name=\'occluded-checkbox\']').on('switchChange.bootstrapSwitch'
        , function(ignoredEvent, ignoredState) {
          if (currentBbox !== -1 && typeof(currentBbox) !== 'undefined') {
            rectDict[currentBbox.id].occluded = $(this).prop('checked');
            bboxLabeling.highlight(currentBbox);
          }
        });

    $('[name=\'truncated-checkbox\']').on('switchChange.bootstrapSwitch'
        , function(ignoredEvent, ignoredState) {
          if (currentBbox !== -1
              && typeof(currentBbox) !== 'undefined') {
            rectDict[currentBbox.id].truncated
                = $(this).prop('checked');
            bboxLabeling.highlight(currentBbox);
          }
        });

    $('#radios :input').change(function() {
      if (currentBbox !== -1 && typeof(currentBbox) !== 'undefined') {
        rectDict[currentBbox.id].traffic_light_color =
            $('input[type=\'radio\']:checked').attr('id');
        bboxLabeling.highlight(currentBbox);
      }
    });

    $(document).on('keydown', function(e) {
      // keyboard shortcut for delete
      if (e.which === 8 || e.which === 46) {
        if (currentBbox !== -1
            && typeof(currentBbox) !== 'undefined') {
          currentBbox.removeBox();
        }
        state = 'free';
        $('#toolbox').css('background-color', '#DCDCDC');
        currentBbox = -1;
        rect = -1;
      }
      // keyboard shortcut for hiding labels
      if (e.keyCode === 72) {
        if (!hideLabels) {
          hideLabels = true;
        } else {
          hideLabels = false;
        }
        bboxLabeling.image_canvas.css('cursor', 'crosshair');
        ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
        ctx.setLineDash([]);
        for (let key in rectDict) {
          if (rectDict[key].hasOwnProperty('num')) {
            let cur = rectDict[key];
            cur.drawBox();
            cur.drawHiddenBox();
            cur.drawTag();
          }
        }
      }
      // "A" key used for checking occluded box
      if (e.keyCode === 65) {
        if (currentBbox !== -1
            && typeof(currentBbox) !== 'undefined') {
          $('[name=\'occluded-checkbox\']').trigger('click');
          rectDict[currentBbox.id].occluded
              = $('[name=\'occluded-checkbox\']').prop('checked');
          bboxLabeling.highlight(currentBbox);
        }
      }

      // "E" key used for checking truncated box
      if (e.keyCode === 83) {
        if (currentBbox !== -1
            && typeof(currentBbox) !== 'undefined') {
          $('[name=\'truncated-checkbox\']').trigger('click');
          rectDict[currentBbox.id].truncated
              = $('[name=\'truncated-checkbox\']').prop('checked');
          bboxLabeling.highlight(currentBbox);
        }
      }
    });

    $('#remove_btn').click(function() {
      if (currentBbox !== -1 && typeof(currentBbox) !== 'undefined') {
        currentBbox.removeBox();
      }
      state = 'free';
      $('#toolbox').css('background-color', '#DCDCDC');
      currentBbox = -1;
      rect = -1;
    });

    $(document).on('mousemove', '#image_canvas', function(e) {
      // Full-canvas crosshair mouse cursor
      let cH = $('#crosshair-h');
      let cV = $('#crosshair-v');
      $('.hair').show();
      let x = e.clientX;
      let y = e.clientY;
      cH.css('top', Math.max(y,
          mainCanvas.getBoundingClientRect().top));
      cH.css('left', mainCanvas.getBoundingClientRect().left);
      cH.css('width', imageCanvasWidth);

      cV.css('right', mainCanvas.getBoundingClientRect().right);
      cV.css('left', Math.max(x,
          mainCanvas.getBoundingClientRect().left));
      cV.css('height', imageCanvasHeight);

      offsetLeft = mainCanvas.getBoundingClientRect().left;
      offsetTop = mainCanvas.getBoundingClientRect().top;

      if (state === 'hover_resize' || state === 'select_resize') {
        if (rect !== -1 && typeof(rect) !== 'undefined') {
          let mousePos = point(e.clientX - offsetLeft,
              e.clientY - offsetTop);
          if (currentHandle >= 0 && currentHandle <= 7) {
            dragHandle[currentHandle](rect, mousePos);
          }
          ctx.clearRect(0, 0
              , imageCanvasWidth, imageCanvasHeight);

          ctx.globalAlpha = 0.5;
          ctx.setLineDash([]);
          for (let key in rectDict) {
            if (key !== rect.id.toString()) {
              let cur = rectDict[key];
              cur.drawBox();
              cur.drawTag();
            }
          }
          ctx.globalAlpha = 1.0;
          ctx.setLineDash([3]);
          rect.drawBox();
          if (currentHandle >= 0 && currentHandle <= 7) {
            ctx.setLineDash([]);
            rect.drawHandle(currentHandle);
          }
        }
      } else if (state === 'draw') {
        rect.update(e.clientX, e.clientY);
      } else if (state === 'select') {
        bboxLabeling.highlight(currentBbox);
      } else {
        // hover on
        mousePos = point(e.clientX - offsetLeft,
            e.clientY - offsetTop);
        previousHandle = currentHandle;

        let returnValue;
        returnValue = bboxLabeling.getSelectedBbox(mousePos);
        currentHandle = returnValue[0];
        selectedBbox = returnValue[1];

        if (selectedBbox !== -1
            && typeof(selectedBbox) !== 'undefined') {
          if (currentHandle >= 0 && currentHandle <= 7) {
            let handlePos
                = bboxHandles[currentHandle](selectedBbox);
            if (dist(mousePos, handlePos)
                < HIDDEN_HANDLE_RADIUS - 2) {
              selectedBbox.drawHandle(currentHandle);
            }
          } else if (currentHandle === 8) {
            bboxLabeling.image_canvas.css('cursor', 'pointer');
          }
        }

        if (currentHandle !== previousHandle) {
          bboxLabeling.image_canvas.css('cursor', 'crosshair');
          ctx.clearRect(0, 0
              , imageCanvasWidth, imageCanvasHeight);
          ctx.setLineDash([]);
          for (let key in rectDict) {
            if (rectDict[key].hasOwnProperty('num')) {
              let cur = rectDict[key];
              cur.drawBox();
              cur.drawHiddenBox();
              cur.drawTag();
            }
          }
        }
      }
    });

    $(document).on('mousedown', '#image_canvas', function(e) {
      offsetLeft = mainCanvas.getBoundingClientRect().left;
      offsetTop = mainCanvas.getBoundingClientRect().top;
      let mousePos = point(e.clientX - offsetLeft
          , e.clientY - offsetTop);
      let returnValue;
      returnValue = bboxLabeling.getSelectedBbox(mousePos);
      currentHandle = returnValue[0];
      selectedBbox = returnValue[1];
      if (currentHandle >= 0 && currentHandle <= 7) {
        rect = selectedBbox;
        addEvent('resize bbox', rect.id, {
          x1: parseFloat((Math.min(rect.x, rect.x + rect.w)
              / ratio).toFixed(6)),
          y1: parseFloat((Math.min(rect.y, rect.y + rect.h)
              / ratio).toFixed(6)),
          x2: parseFloat((Math.max(rect.x, rect.x + rect.w)
              / ratio).toFixed(6)),
          y2: parseFloat((Math.max(rect.y, rect.y + rect.h)
              / ratio).toFixed(6)),
        });
        if (state === 'select') {
          state = 'select_resize';
        } else {
          state = 'hover_resize';
        }
      } else if (currentHandle === 8) {
        currentBbox = selectedBbox;
        rect = selectedBbox;
        state = 'select';
        bboxLabeling.highlight(currentBbox);
        $('#category_select').prop('selectedIndex',
            assignment.category.indexOf(rectDict[currentBbox.id].category));
        if (typeof rectDict[currentBbox.id].occluded !== 'undefined'
            && typeof rectDict[currentBbox.id].truncated
            !== 'undefined') {
          if ($('[name=\'occluded-checkbox\']').prop('checked')
              !== rectDict[currentBbox.id].occluded) {
            $('[name=\'occluded-checkbox\']').trigger('click');
          }
          if ($('[name=\'truncated-checkbox\']').prop('checked')
              !== rectDict[currentBbox.id].truncated) {
            $('[name=\'truncated-checkbox\']').trigger('click');
          }
        }
        if (typeof rectDict[currentBbox.id].traffic_light_color
            !== 'undefined') {
          $('input:radio[id=\'' + rectDict[currentBbox.id].
              traffic_light_color + '\']').trigger('click');
        }
      } else {
        // Unselect
        if (state === 'select') {
          state = 'free';
          currentBbox = -1;
          rect = -1;
          bboxLabeling.image_canvas.css('cursor', 'crosshair');
          $('#toolbox').css('background-color', '#DCDCDC');
        }
        // Draw a new bbox
        let catIdx = document.getElementById('category_select').
            selectedIndex;
        let cat = assignment.category[catIdx];
        let occluded = $('[name=\'occluded-checkbox\']').
            prop('checked');
        let truncated = $('[name=\'truncated-checkbox\']').
            prop('checked');
        let color = $('input[type=\'radio\']:checked').attr('id');
        rect = new BBox(cat, Object.keys(rectDict).length,
            [occluded, truncated, color]);
        rect.start(e.clientX, e.clientY);
        state = 'draw';
      }
    });

    $(document).on('mouseup', '#image_canvas', function() {
      rect.finish();
      if (Math.abs(rect.w) <= 7 && Math.abs(rect.h) <= 7) {
        rect.removeBox();
        state = 'free';
        $('#toolbox').css('background-color', '#DCDCDC');
      } else {
        currentBbox = rect;
        state = 'select';
        bboxLabeling.highlight(currentBbox);
        $('#category_select').prop('selectedIndex',
            assignment.category.indexOf(
                rectDict[currentBbox.id].category));

        if (typeof rectDict[currentBbox.id].occluded
            !== 'undefined'
            && typeof rectDict[currentBbox.id].truncated
            !== 'undefined') {
          if ($('[name=\'occluded-checkbox\']').prop('checked')
              !== rectDict[currentBbox.id].occluded) {
            $('[name=\'occluded-checkbox\']').trigger('click');
          }
          if ($('[name=\'truncated-checkbox\']').prop('checked')
              !== rectDict[currentBbox.id].truncated) {
            $('[name=\'truncated-checkbox\']').trigger('click');
          }
          if (typeof rectDict[currentBbox.id].traffic_light_color
              !== 'undefined') {
            $('input:radio[id=\'' + rectDict[currentBbox.id].
                traffic_light_color + '\']').trigger('click');
          }
        }
      }
    });
  };

  return BBoxLabeling;
})();

// BBox Class
let BBox;
BBox = (function() {
  /**
   * Summary: To be completed.
   * @param {type} category: Description.
   * @param {type} id: Description.
   * @param {int} attribute: Description.
   */
  function BBox(category, id, attribute) {
    SatLabel.call(this, null, id);
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

  BBox.prototype = Object.create(SatLabel.prototype);

  BBox.prototype.start = function(pageX, pageY) {
    this.x = pageX - mainCanvas.getBoundingClientRect().left;
    this.y = pageY - mainCanvas.getBoundingClientRect().top;
    this.w = 0;
    this.h = 0;
    addEvent('draw bbox', this.id, {
      x1: parseFloat((Math.min(this.x, this.x + this.w) / ratio).
          toFixed(6)),
      y1: parseFloat((Math.min(this.y, this.y + this.h) / ratio).
          toFixed(6)),
      x2: parseFloat((Math.max(this.x, this.x + this.w) / ratio).
          toFixed(6)),
      y2: parseFloat((Math.max(this.y, this.y + this.h) / ratio).
          toFixed(6)),
    });
  };

  BBox.prototype.update = function(pageX, pageY) {
    this.w
        = (pageX - mainCanvas.getBoundingClientRect().left) - this.x;
    this.h = (pageY - mainCanvas.getBoundingClientRect().top) - this.y;

    ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);

    ctx.globalAlpha = 0.5;
    ctx.setLineDash([]);
    for (let key in rectDict) {
      if (rectDict[key]) {
        let cur = rectDict[key];
        cur.drawBox();
        cur.drawTag();
      }
    }
    ctx.globalAlpha = 1.0;
    ctx.setLineDash([3]);
    this.drawBox();
  };

  BBox.prototype.finish = function() {
    addEvent('finish bbox', this.id, {
      x1: parseFloat((Math.min(this.x, this.x + this.w) / ratio).
          toFixed(6)),
      y1: parseFloat((Math.min(this.y, this.y + this.h) / ratio).
          toFixed(6)),
      x2: parseFloat((Math.max(this.x, this.x + this.w) / ratio).
          toFixed(6)),
      y2: parseFloat((Math.max(this.y, this.y + this.h) / ratio).
          toFixed(6)),
    });
    ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
    ghostCtx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);

    ctx.globalAlpha = 0.5;
    ctx.setLineDash([]);
    for (let key in rectDict) {
      if (rectDict[key]) {
        let cur = rectDict[key];
        cur.drawBox();
        cur.drawHiddenBox();
        cur.drawTag();
      }
    }
    ctx.globalAlpha = 1.0;
    this.drawBox();
    this.drawHiddenBox();
    this.drawTag();

    if (this.id === Object.keys(rectDict).length) {
      if (this.category === 'traffic light') {
        numLight = numLight + 1;
      }
      numBbox = numBbox + 1;
    }
    $('#bbox_count').text(numBbox);
    $('#light_count').text(numLight);

    rectDict[this.id] = this;
  };

  BBox.prototype.drawTag = function() {
    if (!hideLabels) {
      if (this.category && Math.abs(this.w) > 7
          && Math.abs(this.h) > 7) {
        let x1 = Math.min(this.x, this.x + this.w);
        let y1 = Math.min(this.y, this.y + this.h);
        ctx.font = '11px Verdana';
        let tagWidth = TAG_WIDTH;
        let words = this.category.split(' ');
        let abbr = words[words.length - 1].substring(0, 3);
        if (this.occluded) {
          abbr += ',' + 'o';
          tagWidth += 9;
        }
        if (this.truncated) {
          abbr += ',' + 't';
          tagWidth += 9;
        }
        if (this.traffic_light_color
            && this.traffic_light_color !== 'none') {
          abbr += ',' + this.traffic_light_color.substring(0, 1);
          tagWidth += 9;
        }
        ctx.fillStyle = this.styleColor();
        ctx.fillRect(x1 - 1, y1 - TAG_HEIGHT, tagWidth,
            TAG_HEIGHT);
        ctx.fillStyle = 'rgb(0, 0, 0)';
        ctx.fillText(abbr, x1 + 1, y1 - 3);
      }
    }
  };

  BBox.prototype.drawBox = function() {
    if (Math.abs(this.w) <= 7 && Math.abs(this.h) <= 7) {
      ctx.strokeStyle = 'rgb(169, 169, 169)';
    } else {
      ctx.strokeStyle = this.styleColor();
    }
    ctx.lineWidth = LINE_WIDTH;
    ctx.strokeRect(this.x, this.y, this.w, this.h);
  };

  BBox.prototype.drawHiddenBox = function() {
    // draw hidden box frame
    ghostCtx.lineWidth = HIDDEN_LINE_WIDTH;
    ghostCtx.strokeStyle = this.hidden_colors(this.id, 8);
    ghostCtx.strokeRect(this.x, this.y, this.w, this.h);

    // draw hidden tag
    if (!hideLabels) {
      let x1 = Math.min(this.x, this.x + this.w);
      let y1 = Math.min(this.y, this.y + this.h);
      ghostCtx.fillStyle = this.hidden_colors(this.id, 8);
      ghostCtx.fillRect(x1 - 1, y1 - TAG_HEIGHT, TAG_WIDTH,
          TAG_HEIGHT);
    }
    // draws eight hidden handles
    for (let i = 0; i < 8; i++) {
      this.drawHiddenHandle(i);
    }
  };

  BBox.prototype.drawHandle = function(index) {
    let handlesSize = HANDLE_RADIUS;
    let posHandle = bboxHandles[index](this);
    ctx.beginPath();
    ctx.arc(posHandle.x, posHandle.y, handlesSize, 0, 2 * Math.PI);
    ctx.fillStyle = this.styleColor();
    ctx.fill();

    ctx.lineWidth = 1;
    ctx.strokeStyle = 'white';
    ctx.stroke();
  };

  BBox.prototype.drawHiddenHandle = function(index) {
    let handlesSize = HIDDEN_HANDLE_RADIUS;
    let posHandle = bboxHandles[index](this);
    ghostCtx.beginPath();
    ghostCtx.arc(posHandle.x, posHandle.y, handlesSize, 0,
        2 * Math.PI);
    ghostCtx.fillStyle = this.hidden_colors(this.id, index);
    ghostCtx.fill();
  };

  BBox.prototype.hidden_colors = function(id, handleIndex) {
    return 'rgb(' + (id + 1) + ',' + (handleIndex + 1) + ',0)';
  };

  BBox.prototype.removeBox = function() {
    addEvent('remove bbox', this.id, {
      x1: parseFloat((Math.min(this.x, this.x + this.w) / ratio).
          toFixed(6)),
      y1: parseFloat((Math.min(this.y, this.y + this.h) / ratio).
          toFixed(6)),
      x2: parseFloat((Math.max(this.x, this.x + this.w) / ratio).
          toFixed(6)),
      y2: parseFloat((Math.max(this.y, this.y + this.h) / ratio).
          toFixed(6)),
    });
    ctx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
    ghostCtx.clearRect(0, 0, imageCanvasWidth, imageCanvasHeight);
    let tempDict = rectDict;
    rectDict = {};
    let i = 0;
    for (let key in tempDict) {
      if (tempDict[key].hasOwnProperty('id')) {
        let temp = tempDict[key];
        if (key !== this.id.toString()) {
          temp.id = i;
          rectDict[i] = temp;
          temp.drawBox();
          temp.drawHiddenBox();
          temp.drawTag();
        }
        i++;
      }
    }
    if (this.category === 'traffic light') {
      numLight = numLight - 1;
    }
    numBbox = numBbox - 1;
    $('#bbox_count').text(numBbox);
    $('#light_count').text(numLight);
  };

  return BBox;
})();
