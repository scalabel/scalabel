/* global addEvent SatImage SatLabel numPoly:true ratio:true
 sourceImage:true LabelChart:true LabelList:true polyLabeling:true
 imageList:true currentIndex:true numPoly:true
*/

(function() {
  let polylist = [];
  let mainCanvas = document.getElementById('main_canvas');
  let ctx = mainCanvas.getContext('2d');
  let hiddenCanvas = document.getElementById('hidden_canvas');
  let hiddenCtx = hiddenCanvas.getContext('2d');
  let tempCanvas = document.getElementById('temp_canvas');
  let tempCtx = tempCanvas.getContext('2d');

  let targetPoly = -1;
  let borderPoly = -1;
  let linkList = [];
  let labelShowtime = 1;
  let getInVertex = false;
  let getInPoly = false;
  let state;
  let styleWidth;
  let styleHeight;
  let ShowMid = true;
  let MaxID = 0; // number of objects created so far
  let canvasScale = 1;
  let pointList = []; // points in the buffer during quick-draw
  let showall = true;
  let drawOn = false;

  let MIN_DIS = 6;
  let VERTEX = 1;
  let MID_VERTEX = 2;
  let MAX_DIS = 10000;
  let VERTEX_FILL = 'rgb(100,200,100)';
  let MID_VERTEX_FILL = 'rgb(200,100,100)';
  let MID_CONTROL_POINT = 'rgba(233, 133, 166, 0.7)';
  let BEZIER_COLOR = 'rgba(200,200,100,0.7)';
  let SELECT_COLOR = 'rgba(100,0,100,0.5)';
  let ALPHA = 0.3;

  /**
   * Summary: Calculate the distance between two points.
   * @param {type} point1: the first point.
   * @param {type} point2: the second point.
   * @return {int} distance.
   */
  function calcuDis(point1, point2) {
    return Math.abs(point1[0] - point2[0]) +
        Math.abs(point1[1] - point2[1]);
  }

  /**
   * Summary: compare v1 and v2.
   * @param {type} v1: Description.
   * @param {type} v2: Description.
   * @return {int} 1 if v1 < v2, 1 if v1 > v2, 0 otherwise.
   */
  function compare(v1, v2) {
    if (v1 < v2) return -1;
    else if (v1 > v2) return 1;
    else return 0;
  }

  /**
   * Summary: To be completed.
   * @param {type} poly: Description.
   */
  function deletePolyfromPolylist(poly) {
    let index = poly.listIndex;
    polylist.splice(index, 1);
    addEvent('deletePoly', poly.id);
    numPoly -= 1;
    $('#poly_count').text(numPoly);

    for (let i = 0; i < polylist.length; i++) {
      if (i >= index) {
        polylist[i].listIndex--;
      }
      for (let key in polylist[i].sameIdList) {
        if (polylist[i].sameIdList[key] > index) {
          polylist[i].sameIdList[key]--;
        } else if (polylist[i].sameIdList[key] === index) {
          polylist[i].sameIdList.splice(index, 1);
        }
      }
      for (let key in polylist[i].insideList) {
        if (polylist[i].insideList[key] > index) {
          polylist[i].insideList[key]--;
        } else if (polylist[i].insideList[key] === index) {
          polylist[i].insideList.splice(index, 1);
        }
      }
      for (let key in polylist[i].outsideList) {
        if (polylist[i].outsideList[key] > index) {
          polylist[i].outsideList[key]--;
        } else if (polylist[i].outsideList[key] === index) {
          polylist[i].outsideList.splice(index, 1);
        }
      }
    }
  }

  /**
   * Summary: To be completed.
   *
   */
  function resizeCanvas() {
    ratio = parseFloat(window.innerWidth / (1.35 * sourceImage.width));
    if (parseFloat(window.innerHeight /
            (1.35 * sourceImage.height)) < ratio) {
      ratio = parseFloat(window.innerHeight /
          (1.35 * sourceImage.height));
    }
    ratio = parseFloat(ratio.toFixed(6));

    styleWidth = Math.round(sourceImage.width * ratio);
    styleHeight = Math.round(sourceImage.height * ratio);

    mainCanvas.style.width = styleWidth + 'px';
    mainCanvas.style.height = styleHeight + 'px';
    hiddenCanvas.style.width = styleWidth + 'px';
    hiddenCanvas.style.height = styleHeight + 'px';
    tempCanvas.style.width = styleWidth + 'px';
    tempCanvas.style.height = styleHeight + 'px';

    mainCanvas.height = styleHeight * 2;
    mainCanvas.width = styleWidth * 2;
    hiddenCanvas.height = styleHeight * 2;
    hiddenCanvas.width = styleWidth * 2;
    tempCanvas.height = styleHeight * 2;
    tempCanvas.width = styleWidth * 2;

    ctx.scale(2, 2);
    tempCtx.scale(2, 2);
    hiddenCtx.scale(2, 2);
  }

  /**
   * Summary: To be completed.
   * @param {type} poly: Description.
   * @return {type} Description.
   */
  function calcuCenter(poly) {
    let x = 0;
    let y = 0;
    for (let i = 0; i < poly.num; i++) {
      x += poly.p[i][0];
      y += poly.p[i][1];
    }
    return [Math.round(x / poly.num), Math.round(y / poly.num)];
  }

  /**
   * Summary: To be completed.
   * @param {type} point: Description.
   * @param {type} cnt: current vertice number.
   * @param {type} poly: Description.
   * @param {type} context: Description.
   */
  function drawLine(point, cnt, poly, context) {
    context.lineWidth = 3 / canvasScale;
    context.strokeStyle = poly.styleColor(1);
    context.beginPath();
    context.moveTo(poly.p[0][0], poly.p[0][1]);
    for (let i = 1; i < cnt; i++) {
      context.lineTo(poly.p[i][0], poly.p[i][1]);
    }
    context.lineTo(point[0][0], point[0][1]);
    for (let i = 1; i < point.length; i++) {
      context.lineTo(point[i][0], point[i][1]);
    }
    context.lineTo(poly.p[cnt][0], poly.p[cnt][1]);
    context.closePath();
    context.stroke();
  }

  /**
   * Summary: To be completed.
   *
   */
  function incHandler() {
    canvasScale = canvasScale << 1;
    if (canvasScale === 2) {
      $('#decrease_btn').attr('disabled', false);
    }
    if (canvasScale === 4) {
      $('#increase_btn').attr('disabled', true);
    }

    styleWidth = styleWidth << 1;
    styleHeight = styleHeight << 1;

    mainCanvas.style.width = styleWidth + 'px';
    mainCanvas.style.height = styleHeight + 'px';
    tempCanvas.style.width = styleWidth + 'px';
    tempCanvas.style.height = styleHeight + 'px';
    hiddenCanvas.style.width = styleWidth + 'px';
    hiddenCanvas.style.height = styleHeight + 'px';
    // 1, 2: 2
    // 4: 4
    if (canvasScale > 2) { // 4
      mainCanvas.height = mainCanvas.height << 1;
      mainCanvas.width = mainCanvas.width << 1;
      tempCanvas.height = tempCanvas.height << 1;
      tempCanvas.width = tempCanvas.width << 1;
      hiddenCanvas.height = hiddenCanvas.height << 1;
      hiddenCanvas.width = hiddenCanvas.width << 1;

      ctx.scale(canvasScale, canvasScale);
      hiddenCtx.scale(canvasScale, canvasScale);
      tempCtx.scale(canvasScale, canvasScale);
    }

    polyLabeling.redraw();
    polyLabeling.hidden_redraw();
  }

  /**
   * Summary: To be completed.
   *
   */
  function decHandler() {
    canvasScale = canvasScale >> 1;

    if (canvasScale === 1) {
      $('#decrease_btn').attr('disabled', true);
    }
    if (canvasScale === 2) {
      $('#increase_btn').attr('disabled', false);
    }

    styleWidth = styleWidth >> 1;
    styleHeight = styleHeight >> 1;
    mainCanvas.style.width = styleWidth + 'px';
    mainCanvas.style.height = styleHeight + 'px';
    tempCanvas.style.width = styleWidth + 'px';
    tempCanvas.style.height = styleHeight + 'px';
    hiddenCanvas.style.width = styleWidth + 'px';
    hiddenCanvas.style.height = styleHeight + 'px';

    if (canvasScale >= 2) { // 4
      mainCanvas.height = mainCanvas.height >> 1;
      mainCanvas.width = mainCanvas.width >> 1;
      tempCanvas.height = tempCanvas.height >> 1;
      tempCanvas.width = tempCanvas.width >> 1;

      hiddenCanvas.height = hiddenCanvas.height >> 1;
      hiddenCanvas.width = hiddenCanvas.width >> 1;

      ctx.scale(canvasScale, canvasScale);
      hiddenCtx.scale(canvasScale, canvasScale);
      tempCtx.scale(canvasScale, canvasScale);
    }

    polyLabeling.redraw();
    polyLabeling.hidden_redraw();
  }

  this.PolyLabeling = (function() {
    /**
     * Summary: To be completed.
     * @param {type} options: Description.
     * @return {type}: Description.
     */
    function PolyLabeling(options) {
      SatImage.call(this, options);
      this.options = options;

      $('#main_canvas').css({
        'background-image': 'url(\'' + this.options.url + '\')',
        'cursor': 'crosshair',
      });
      // Start listening to events happening in the main panel
      return this.eventController();
    }

    PolyLabeling.prototype = Object.create(SatImage.prototype);

    PolyLabeling.prototype.updateImage = function(url) {
      sourceImage = new Image();
      sourceImage.onload = function() {
        resizeCanvas();
        polyLabeling.replay();
      };
      sourceImage.src = url;
      this.options.url = url;
      let polyLabeling = this;
      $('#main_canvas').css({
        'background-image': 'url(\'' + url + '\')',
        'cursor': 'crosshair',
      });
    };

    PolyLabeling.prototype.drawSelectVertex = function(mode, pos) {
      let color = VERTEX_FILL;
      if (mode === MID_VERTEX) color = MID_VERTEX_FILL;
      tempCtx.beginPath();
      tempCtx.arc(pos[0], pos[1], 6 / canvasScale, 0,
          2 * Math.PI, false);
      tempCtx.closePath();
      tempCtx.fillStyle = color;
      tempCtx.fill();
    };

    PolyLabeling.prototype.linkPolyHandler = function() {
      let id;
      let targetPoly;
      let poly = [];
      let sameIDList = [];
      if (linkList.length < 2) return;
      if (linkList.length > 0) {
        targetPoly = linkList[0];
        id = targetPoly.id;
        for (let i = 1; i < linkList.length; i++) {
          poly = linkList[i];
          if (poly.category === targetPoly.category) {
            if (id > poly.id) {
              id = poly.id;
            }
          } else {
            alert('can not link poly with different categories!');
            return;
          }
        }
        for (let i = 0; i < linkList.length; i++) {
          sameIDList.push(linkList[i].listIndex);
          for (let j = 0; j < linkList[i].sameIdList.length; j++) {
            if (sameIDList.indexOf(linkList[i].sameIdList[j])
                === -1) {
              sameIDList.push(linkList[i].sameIdList[j]);
            }
          }
        }
        for (let i = 0; i < sameIDList.length; i++) {
          polylist[sameIDList[i]].id = id;
          polylist[sameIDList[i]].sameIdList = sameIDList;
        }
      }
      polyLabeling.redraw();
    };

    PolyLabeling.prototype.submitLabels = function() {
      this.output_labels = [];
      let tmp = [];
      let poly;
      let output;

      for (let key in polylist) {
        if (polylist[key].hasOwnProperty('num')) {
          poly = polylist[key];
          if (ratio) {
            tmp = [];
            for (let i = 0; i < poly.num; i++) {
              tmp[i] =
                  [
                    parseFloat((poly.p[i][0] / ratio).toFixed(6)),
                    parseFloat((poly.p[i][1] / ratio).toFixed(6))];
            }
            // console.log(ratio);
          }
          output = {
            id: poly.id.toString(),
            category: poly.category,
            position: {
              listIndex: poly.listIndex,
              p: tmp,
              sameIdList: poly.sameIdList,
              insideList: poly.insideList,
              outsideList: poly.outsideList,
              num: poly.num,
              BezierOffset: poly.BezierOffset,
              beziernum: poly.beziernum,
            },
          };
          this.output_labels.push(output);
        }
      }
    };

    PolyLabeling.prototype.redraw = function(exception) {
      let poly;
      let pos;

      ctx.clearRect(0, 0,
          mainCanvas.width, mainCanvas.height);

      if (polylist && showall == true) {
        for (let key in polylist) {
          if (exception && polylist[key] == exception) {
            continue;
          } else {
            poly = polylist[key];
            if (poly.outsideList.length == 0) {
              poly.drawPoly(poly.num, ctx);
            }
            if (labelShowtime) {
              pos = calcuCenter(poly);
              poly.fillLabel(pos, ctx);
            }
          }
        }
      }

      if (targetPoly !== -1 && state === 'select') {
        targetPoly.drawPoly(targetPoly.num, ctx, true);
      }

      if (linkList.length !== 0 && state === 'link') {
        for (let key in linkList) {
          if (linkList[key].hasOwnProperty('num')) {
            linkList[key].drawPoly(linkList[key].num, ctx, true);
          }
        }
      }
    };

    PolyLabeling.prototype.hidden_redraw = function() {
      let poly;
      hiddenCtx.clearRect(0, 0,
          hiddenCanvas.width, hiddenCanvas.height);
      if (polylist) {
        for (let key in polylist) {
          if (polylist[key].hasOwnProperty('num')) {
            poly = polylist[key];
            poly.drawHiddenPoly(poly.num);
          }
        }
      }
    };

    PolyLabeling.prototype.clearGlobals = function() {
      getInVertex = false;
      getInPoly = false;
      this.resize = false;
      targetPoly = -1;
      drawOn = false;
      $('#inside_btn').text('On Top');
      state = 'free';
      linkList = [];
      // $("#submit_btn").attr("disabled", false);
    };

    PolyLabeling.prototype.clearAll = function() {
      ctx.clearRect(0, 0,
          mainCanvas.width, mainCanvas.height);
      hiddenCtx.clearRect(0, 0,
          hiddenCanvas.width, hiddenCanvas.height);

      polylist = [];
      MaxID = 0;
      labelShowtime = 1;
      canvasScale = 1;
      $('#label_btn').text('Hide Label (L)');
      // magnify = false;
      borderPoly = -1;
      $('#decrease_btn').attr('disabled', true);
      this.clearGlobals();
      showall = true;
      $('#show_all').text('Hide All (H)');
    };

    PolyLabeling.prototype.replay = function() {
      let polyLabeling;
      polyLabeling = this;

      // console.log(ratio);
      if (typeof(ratio) === 'undefined' || ratio < 0) {
        alert('Error when preloading. Please refresh page.');
        return;
      }
      polylist = [];
      let poly = -1;
      let label = -1;
      let maxId = -1;
      let labelList = imageList[currentIndex].labels;

      ctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
      hiddenCtx.clearRect(0, 0,
          hiddenCanvas.width, hiddenCanvas.height);
      if (labelList) {
        for (let key in labelList) {
          if (!labelList.hasOwnProperty(key)) continue;
          label = labelList[key];
          poly = new Poly(label.category, parseInt(label.id));
          poly.bbox = [10000, 10000, -1, -1];
          poly.listIndex = label.position.listIndex;
          if (poly.id > maxId) maxId = poly.id;
          for (let i = 0; i < label.position.num; i++) {
            poly.p.push([
              parseFloat((label.position.p[i][0] * ratio).
                  toFixed(6)),
              parseFloat((label.position.p[i][1] * ratio).
                  toFixed(6))]);
            if (poly.p[i][0] < poly.bbox[0]) {
              poly.bbox[0] = poly.p[i][0];
            }
            if (poly.p[i][0] > poly.bbox[2]) {
              poly.bbox[2] = poly.p[i][0];
            }
            if (poly.p[i][1] < poly.bbox[1]) {
              poly.bbox[1] = poly.p[i][1];
            }
            if (poly.p[i][1] > poly.bbox[3]) {
              poly.bbox[3] = poly.p[i][1];
            }

            if (i > 0) {
              poly.hidden_p[i - 1] =
                  [
                    (poly.p[i][0] + poly.p[i - 1][0]) / 2,
                    (poly.p[i][1] + poly.p[i - 1][1]) / 2];
            }
          }
          if (label.position.num > 0) {
            poly.hidden_p[label.position.num - 1] =
              [
                (poly.p[label.position.num - 1][0] + poly.p[0][0]) / 2,
                (poly.p[label.position.num - 1][1] + poly.p[0][1]) / 2];
          }

          poly.num = label.position.num;
          poly.BezierOffset = label.position.BezierOffset;
          poly.beziernum = label.position.beziernum;
          if (label.position.sameIdList.length == 0) {
            poly.sameIdList = label.position.sameIdList;
          } else if (
              typeof(label.position.sameIdList[0]) !== 'undefined') {
            poly.sameIdList = label.position.sameIdList;
          }

          if (typeof(label.position.insideList) !== 'undefined') {
            poly.insideList = label.position.insideList;
          }
          if (typeof(label.position.outsideList) !== 'undefined') {
            poly.outsideList = label.position.outsideList;
          }
          poly.computeDegree();
          polylist[poly.listIndex] = poly;
        }
      }
      polyLabeling.redraw();
      polyLabeling.hidden_redraw();
      MaxID = maxId + 1;
    };
    PolyLabeling.prototype.selectPoly = function(clickX, clickY) {
      let r = 2;
      let pixelData;
      if (canvasScale > 2) r = canvasScale;
      pixelData =
          hiddenCtx.getImageData(r * clickX, r * clickY, 1, 1).data;
      let selectedPoly = -1;
      let id;
      if (pixelData[0] != 0 && pixelData[3] == 255) {
        id = pixelData[0] - 1;
        selectedPoly = polylist[id];
      }
      return selectedPoly;
    };
    PolyLabeling.prototype.selectPolyVertex = function(clickX, clickY) {
      let r = 2;
      let pixelData;
      if (canvasScale > 2) r = canvasScale;
      pixelData =
          hiddenCtx.getImageData(r * clickX, r * clickY, 1, 1).data;

      let selectedVertex = -1;
      if (pixelData[0] === 0) return selectedVertex;
      if (pixelData[2] === 0 && pixelData[1] === 0) {
        return selectedVertex;
      }

      // mid
      let y = pixelData[1];
      let z = pixelData[2];

      selectedVertex = ((y << 8) | z) - 1;
      // console.log(selectedVertex);
      return selectedVertex;
    };

    PolyLabeling.prototype.eventController = function() {
      let polyLabeling;
      let poly;
      let selectedPoly;
      let cnt = 0;
      let IfGenarate = false;
      let minPoly;
      let minPoint;
      let minMidPoint;
      let lastPoly = -1;
      let curpos = [];
      let selectCount = 0;
      let borderPoint = [];
      let pointList1 = []; // clockwise
      let pointList2 = []; // counter-clockwise
      let interval = null;
      let enable = null;
      polyLabeling = this;
      state = 'free';

      /**
       * Summary: To be completed.
       * @param {type} e: Description.
       * @return {type}: Description.
       */
      function computePosition(e) {
        let bbox = mainCanvas.getBoundingClientRect();
        let mouseX = parseFloat(((e.clientX - bbox.left)
            * (styleWidth / bbox.width)).toFixed(6));
        let mouseY = parseFloat(((e.clientY - bbox.top
        ) * (styleHeight / bbox.height)).toFixed(6));

        mouseX = parseFloat((mouseX / canvasScale).toFixed(6));
        mouseY = parseFloat((mouseY / canvasScale).toFixed(6));
        if (state == 'free' || state == 'select') {
          if (mouseX < 0 || mouseY < 0 ||
              mouseX > styleWidth || mouseY > styleHeight) {
            return [-1, -1];
          }
        } else {
          if (mouseX < 0) mouseX = 0.000000;
          if (mouseY < 0) mouseY = 0.000000;
          if (mouseX > styleWidth / canvasScale) {
            mouseX = styleWidth / canvasScale;
          }
          if (mouseY > styleHeight / canvasScale) {
            mouseY = styleHeight / canvasScale;
          }
        }
        return [mouseX, mouseY];
      }

      /**
       * Summary: To be completed.
       *
       */
      function clearBorderInfo() {
        pointList = [];
        pointList1 = [];
        pointList2 = [];
        if (borderPoly !== -1) {
          tempCtx.clearRect(borderPoly.bbox[0] - 4,
              borderPoly.bbox[1] - 4,
              borderPoly.bbox[2] - borderPoly.bbox[0] + 8,
              borderPoly.bbox[3] - borderPoly.bbox[1] + 8);
        }
        borderPoly = -1;
        selectCount = 0;
        borderPoint = [];
      }

      /**
       * Summary: To be completed.
       * @param {type} e: Description.
       * @return {type}: Description.
       */
      function mouseDownHandler(e) {
        let Clickpos = computePosition(e);
        let clickX = Clickpos[0];
        let clickY = Clickpos[1];
        if (clickX < 0 || clickY < 0) return;

        if (e.which === 1) {
          if (state == 'free') cnt = 0;
          if (state == 'select'
              && polyLabeling.resize == 0) {
            polyLabeling.clearGlobals();
            polyLabeling.redraw();
            cnt = 0;
            state = 'free';
          }
          if (state == 'select') {
            if (polyLabeling.resize > 0) {
              state = 'select_resize';
            }
          } else if (state === 'quick_draw') {
            if (polyLabeling.resize == 0) {
              if (pointList.length > 0) {// update border
                for (let i = 0; i < pointList.length; i++) {
                  poly.update(pointList[i][0],
                      pointList[i][1], cnt);
                  cnt++;
                  poly.num++;
                }
              }
              clearBorderInfo();
              $('#toolhead').css('background-color', '#00F0E0');
              getInVertex = false;
              getInPoly = false;

              poly.update(clickX, clickY, cnt);
              cnt++;
              poly.num++;
            }
          } else if (state === 'link') {
            if (polyLabeling.resize == 0) {
              selectedPoly =
                  polyLabeling.selectPoly(clickX, clickY);
              if (selectedPoly != -1) {
                let idx = linkList.indexOf(selectedPoly);
                if (idx === -1) {
                  linkList.push(selectedPoly);
                  selectedPoly.drawPoly(selectedPoly.num,
                      ctx, true);
                } else {
                  linkList.splice(idx, 1);
                  polyLabeling.redraw();
                }
              }
            }
          } else if (cnt === 0 && state == 'free') {
            if (drawOn == false) {
              targetPoly = -1;
            }
            if (drawOn == true) {
              polyLabeling.resize = 0;
            }
            selectedPoly = polyLabeling.selectPoly(clickX, clickY);
            if (polyLabeling.resize == 0) {
              if (drawOn == true && selectedPoly !== targetPoly) {
                alert('please click inside of the'
                    + 'selected region!');
              } else {
                state = 'draw';
                let catIdx = document.getElementById(
                    'category_select').selectedIndex;
                let nameIdx = document.getElementById(
                    'name_select').selectedIndex;
                let cate = LabelChart[catIdx][nameIdx];
                poly = new Poly(cate, MaxID);
                poly.listIndex = polylist.length;
                getInVertex = false;
                getInPoly = false;

                poly.update(clickX, clickY, 0);
                cnt++;
                poly.num++;
              }
            } else {
              state = 'resize';
            }
          } else {
            if (state == 'finish_draw' ||
                state == 'finish_quick_draw') { // closing
              if (state == 'finish_quick_draw') {
                if (pointList.length > 0) {// update border
                  for (let i = 0; i < pointList.length; i++) {
                    poly.update(pointList[i][0],
                        pointList[i][1], cnt);
                    cnt++;
                    poly.num++;
                  }
                }
                clearBorderInfo();
                ShowMid = true;
                $('#toolhead').css('background-color',
                    '#DDDDDD');
              }
              window.addEventListener('dblclick',
                  DoubleClick, false);
              poly.num = cnt;
              poly.hidden_p[cnt - 1] =
                  [
                    (poly.p[cnt - 1][0] +
                        poly.p[0][0]) / 2,
                    (poly.p[cnt - 1][1] +
                        poly.p[0][1]) / 2];
              poly.p.splice(cnt, 1);
              poly.hidden_p.splice(cnt, 1);
              poly.degree.splice(cnt, 1);

              if (poly.num == 1) poly.hidden_p = [];

              cnt = 0;

              polylist.push(poly);

              MaxID += 1;
              numPoly += 1;
              $('#poly_count').text(numPoly);

              tempCtx.clearRect(0, 0, tempCanvas.width,
                  tempCanvas.height);

              if (drawOn == true) {
                targetPoly.insideList.push(poly.listIndex);
                poly.outsideList.push(targetPoly.listIndex);
                drawOn = false;
                $('#inside_btn').text('On Top');
                polyLabeling.redraw();
                targetPoly.drawPoly(targetPoly.num, ctx);
              }
              poly.drawPoly(poly.num, ctx);
              polyLabeling.clearGlobals();

              if (labelShowtime) {
                let center = calcuCenter(poly);
                poly.fillLabel(center, ctx);
              }
              poly.recomputeBbox();
              poly.drawHiddenPoly(poly.num);
            } else if (state == 'draw') {
              let dis = calcuDis(poly.p[cnt - 1],
                  [clickX, clickY]);
              selectedPoly =
                  polyLabeling.selectPoly(clickX, clickY);
              if (dis > 2 / canvasScale) {
                if (selectedPoly !== targetPoly
                    && drawOn == true) {
                  alert('please click inside of the'
                      + ' selected region!');
                } else {
                  poly.update(clickX, clickY, cnt);
                  cnt++;
                  poly.num++;
                  window.removeEventListener('dblclick',
                      DoubleClick, false);
                }
              }
            }
          }
        }
        return true;
      }

      /**
       * Summary: To be completed.
       * @param {type} e: Description.
       */
      function mouseDown(e) {
        // do nothing if the user tries to click on the scroll bar
        if (e.offsetX > e.target.clientWidth
            || e.offsetY > e.target.clientHeight) {
          return;
        }

        if (polyLabeling.resize > 0) {
          mouseDownHandler(e);
          return;
        }

        let pos = computePosition(e);
        if (pos[0] < 0 || pos[1] < 0) return;

        let selectedPoly = polyLabeling.selectPoly(pos[0], pos[1]);

        if (state == 'select') {
          // new draw or deselect/change selection
          clearTimeout(interval);
          interval = setTimeout(function() {
            mouseDownHandler(e);
          }, 300);
        } else if (state == 'free' && selectedPoly !== -1) {
          // not background, judge if is dbclick
          clearTimeout(interval);
          interval = setTimeout(function() {
            mouseDownHandler(e);
          }, 300);
        } else {
          mouseDownHandler(e);
        }
      }

      /**
       * Summary: To be completed.
       * @param {type} mouseX: Description.
       * @param {type} mouseY: Description.
       */
      function drawPolyHandler(mouseX, mouseY) {
        poly.p[cnt] = [mouseX, mouseY];
        poly.num = cnt;
        curpos = [mouseX, mouseY];
        let currentDis = MIN_DIS / canvasScale;

        if (calcuDis(curpos, poly.p[0]) <= currentDis && (cnt > 1)) {
          if (state == 'draw') {
            state = 'finish_draw';
          } else if (state == 'quick_draw') {
            state = 'finish_quick_draw';
          }
          tempCtx.beginPath();
          tempCtx.arc(poly.p[0][0], poly.p[0][1],
              6 / canvasScale, 0, 2 * Math.PI, false);
          tempCtx.fillStyle = poly.styleColor(1);
          tempCtx.closePath();
          tempCtx.fill();
          tempCtx.beginPath();
          tempCtx.arc(poly.p[0][0], poly.p[0][1],
              4 / canvasScale, 0, 2 * Math.PI, false);
          tempCtx.fillStyle = VERTEX_FILL;
          tempCtx.closePath();
          tempCtx.fill();
        } else {
          // keep drawing
          if (state === 'draw') {
            let tmpbbox = poly.bbox;
            if (mouseX < poly.bbox[0]) tmpbbox[0] = mouseX;
            if (mouseX > poly.bbox[2]) tmpbbox[2] = mouseX;

            if (mouseY < poly.bbox[1]) tmpbbox[1] = mouseY;
            if (mouseY > poly.bbox[3]) tmpbbox[3] = mouseY;
            if (tmpbbox[0] < 4) tmpbbox[0] = 4;
            if (tmpbbox[1] < 4) tmpbbox[1] = 4;

            tempCtx.clearRect(tmpbbox[0] - 4, tmpbbox[1] - 4,
                tmpbbox[2] - tmpbbox[0] + 8, tmpbbox[3] - tmpbbox[1] + 8);
            // tempCtx.strokeRect(tmpbbox[0]-4, tmpbbox[1]-4,
            // tmpbbox[2]-tmpbbox[0] + 2, tmpbbox[3]-tmpbbox[1]+2);
            poly.drawPoly(cnt + 1, tempCtx);
          }
          if (cnt > 0) {
            tempCtx.beginPath();
            tempCtx.arc(poly.p[0][0], poly.p[0][1],
                4 / canvasScale, 0, 2 * Math.PI, false);
            tempCtx.closePath();
            tempCtx.fillStyle = poly.styleColor(0.7);
            tempCtx.fill();
          }
        }
      }

      /**
       * Summary: To be completed.
       * @param {type} mouseX: Description.
       * @param {type} mouseY: Description.
       */
      function mousemoveHandler(mouseX, mouseY) {
        if (minPoly !== -1) {
          if (getInPoly === false) {
            minPoly.showControlPoints(minPoly.num, tempCtx);
            lastPoly = minPoly;
            getInPoly = true;
          }
          if (minPoly !== lastPoly) {
            lastPoly.clearControlPoints(lastPoly.num, tempCtx);
            minPoly.showControlPoints(minPoly.num, tempCtx);
            lastPoly = minPoly;
          }

          if (polyLabeling.resize === VERTEX) {
            if (getInVertex === false) {
              if (minPoly.num > minPoint) {
                getInVertex = true;
                polyLabeling.drawSelectVertex(
                    polyLabeling.resize,
                    minPoly.p[minPoint]);
              }
            }
          } else if (polyLabeling.resize === MID_VERTEX) {
            if (getInVertex === false) {
              if (minPoly.num > minMidPoint) {
                getInVertex = true;
                polyLabeling.drawSelectVertex(
                    polyLabeling.resize,
                    minPoly.hidden_p[minMidPoint]);
              }
            }
          } else {
            if (getInVertex === true) {
              getInVertex = false;
              // lastPoly.clearControlPoints(lastPoly.num, tempCtx);
            }
          }
        } else {// undo
          if (getInVertex === true) {
            getInVertex = false;
            polyLabeling.resize = 0;
            lastPoly.clearControlPoints(lastPoly.num, tempCtx);
          }
          if (getInPoly === true) {
            getInPoly = false;
            polyLabeling.resize = 0;
            lastPoly.clearControlPoints(lastPoly.num, tempCtx);
            lastPoly = -1;
          }
        }
        if (state === 'quick_draw') {
          poly.p[cnt] = [mouseX, mouseY];
          if (cnt > 0 || (cnt === 0 && pointList.length > 0)) {
            let tmpbbox = poly.bbox;
            if (mouseX < poly.bbox[0]) tmpbbox[0] = mouseX;
            if (mouseX > poly.bbox[2]) tmpbbox[2] = mouseX;

            if (mouseY < poly.bbox[1]) tmpbbox[1] = mouseY;
            if (mouseY > poly.bbox[3]) tmpbbox[3] = mouseY;
            if (tmpbbox[0] < 4) tmpbbox[0] = 4;
            if (tmpbbox[1] < 4) tmpbbox[1] = 4;

            tempCtx.clearRect(tmpbbox[0] - 4,
                tmpbbox[1] - 4, tmpbbox[2] - tmpbbox[0] + 8,
                tmpbbox[3] - tmpbbox[1] + 8);
            // tempCtx.strokeRect(tmpbbox[0]-4, tmpbbox[1]-4,
            // tmpbbox[2]-tmpbbox[0] + 2,
            // tmpbbox[3]-tmpbbox[1] + 2);
          }
          if (borderPoly !== -1) {
            tempCtx.clearRect(borderPoly.bbox[0] - 4,
                borderPoly.bbox[1] - 4,
                borderPoly.bbox[2] - borderPoly.bbox[0] + 8,
                borderPoly.bbox[3] - borderPoly.bbox[1] + 8);
            // tempCtx.strokeRect(borderPoly.bbox[0]-4,
            // borderPoly.bbox[1]-4,
            // borderPoly.bbox[2] - borderPoly.bbox[0],
            // borderPoly.bbox[3]- borderPoly.bbox[1]);
          }

          if (getInVertex) {
            polyLabeling.drawSelectVertex(polyLabeling.resize,
                minPoly.p[minPoint]);
          }
          if (pointList.length > 0) {
            drawLine(pointList, cnt, poly, tempCtx);
          } else if (cnt > 0) {
            poly.drawPoly(cnt + 1, tempCtx, false, true, false);
          }
          drawPolyHandler(mouseX, mouseY);

          if (borderPoly !== -1) {
            borderPoly.drawPoly(borderPoly.num, tempCtx, true);
          }
        }
      }

      /**
       * Summary: To be completed.
       * @param {type} e: Description.
       */
      function mouseMove(e) {
        curpos = computePosition(e);
        if (curpos[0] < 0 || curpos[1] < 0) {
          if (getInPoly === true) {
            getInPoly = false;
            getInVertex = false;
            if (lastPoly !== -1) {
              lastPoly.clearControlPoints(lastPoly.num,
                  tempCtx);
            }
          }
          return;
        }
        let mouseX = curpos[0];
        let mouseY = curpos[1];
        if (state === 'select' || state === 'free' ||
            state === 'quick_draw') {
          polyLabeling.resize = 0;
          minPoly = polyLabeling.selectPoly(mouseX, mouseY);
          if (showall === false && minPoly !== poly) {
            return;
          }
          if (state !== 'select') {
            minPoint = polyLabeling.selectPolyVertex(mouseX,
                mouseY);
            // console.log(minPoint);
            if (minPoint % 2 === 0) {
              minPoint = minPoint / 2;
              minMidPoint = -1;
            } else if (minPoint % 2 === 1 && minPoint !== -1) {
              minMidPoint = (minPoint - 1) / 2;
              minPoint = -1;
            } else {
              minPoint = -1;
              minMidPoint = -1;
            }
          } else {
            let minDis = MAX_DIS;
            let curDis = [];
            minPoly = targetPoly;
            if (targetPoly !== -1) {
              for (let i = 0; i < targetPoly.num; i++) {
                curDis = [
                  calcuDis(curpos, targetPoly.p[i]),
                  calcuDis(curpos, targetPoly.hidden_p[i])];
                if (curDis[1] < minDis && ShowMid) {
                  minDis = curDis[1];
                  minMidPoint = i;
                  minPoint = -1;
                }
                if (curDis[0] < minDis) {
                  minDis = curDis[0];
                  minPoint = i;
                  minMidPoint = -1;
                }
              }
            }
            if (minDis >= 6 / canvasScale) {
              minMidPoint = -1;
              minPoint = -1;
              minPoly = polyLabeling.selectPoly(mouseX, mouseY);
            }
          }
          if (state === 'quick_draw' && borderPoly !== -1 &&
              selectCount === 1) {
            // the same poly has a higher priority
            let tmpMinPoint;
            let minDis = MAX_DIS;
            let curDis;
            minMidPoint = -1;
            for (let i = 0; i < borderPoly.num; i++) {
              curDis = calcuDis(curpos, borderPoly.p[i]);
              if (curDis < minDis) {
                minDis = curDis;
                tmpMinPoint = i;
              }
            }
            if (minDis < 6 / canvasScale) {
              minPoly = borderPoly;
              minPoint = tmpMinPoint;
            }
          }

          if (!ShowMid) minMidPoint = -1;

          let vdis = MAX_DIS;
          let mdis = MAX_DIS;

          // double check
          // console.log(minPoint, minMidPoint);
          if (minPoint !== -1 && minPoly.num > minPoint) {
            vdis = calcuDis(curpos, minPoly.p[minPoint]);
          }
          if (minMidPoint !== -1 && minPoly.num > minMidPoint) {
            mdis = calcuDis(curpos, minPoly.hidden_p[minMidPoint]);
          }

          if (minPoint !== -1 && minMidPoint === -1
              && vdis < 6 / canvasScale) {
            polyLabeling.resize = VERTEX;
          } else if (minPoint === -1 && minMidPoint !== -1
              && mdis < 6 / canvasScale) {
            polyLabeling.resize = MID_VERTEX;
          } else if (minPoint !== -1 && minMidPoint !== -1) {
            if (vdis < mdis && vdis < 6 / canvasScale) {
              polyLabeling.resize = VERTEX;
            } else if (mdis <= vdis && mdis < 6 / canvasScale) {
              polyLabeling.resize = MID_VERTEX;
            }
          } else {
            polyLabeling.resize = 0;
          }

          if (polyLabeling.resize === MID_VERTEX) {
            if (minPoly.num <= minMidPoint || minPoly.num <= 0) {
              polyLabeling.resize = 0;
            } else if (minPoly.degree[minMidPoint] === 2 ||
                minPoly.degree[(minMidPoint + 1) % minPoly.num]
                === 2) {
              polyLabeling.resize = 0;
            }
          }

          mousemoveHandler(mouseX, mouseY);
        } else if (state === 'draw') {
          drawPolyHandler(mouseX, mouseY);
        } else if (state === 'finish_draw' ||
            state === 'finish_quick_draw') {
          if (calcuDis(curpos, poly.p[0]) > MIN_DIS / canvasScale) {
            poly.p[cnt] = [mouseX, mouseY];
            poly.num = cnt;
            if (state === 'finish_draw') {
              state = 'draw';
            } else if (state === 'finish_quick_draw') {
              state = 'quick_draw';
            }
            poly.drawPoly(cnt + 1, tempCtx);
          }
        } else if (state === 'resize' || state === 'select_resize') {
          selectedPoly = polyLabeling.selectPoly(mouseX, mouseY);
          for (let key in minPoly.insideList) {
            if (selectedPoly.listIndex ===
                minPoly.insideList[key]) {
              return;
            }
          }
          for (let key in minPoly.outsideList) {
            if (selectedPoly.listIndex !== minPoly.outsideList[key]
                && selectedPoly !== minPoly) {
              return;
            }
          }

          getInVertex = false;
          getInPoly = false;
          minPoly.clearControlPoints(minPoly.num, tempCtx);
          let tmpbbox = minPoly.bbox;
          if (mouseX < minPoly.bbox[0]) tmpbbox[0] = mouseX;
          if (mouseX > minPoly.bbox[2]) tmpbbox[2] = mouseX;

          if (mouseY < minPoly.bbox[1]) tmpbbox[1] = mouseY;
          if (mouseY > minPoly.bbox[3]) tmpbbox[3] = mouseY;
          if (tmpbbox[0] < 2) tmpbbox[0] = 2;
          if (tmpbbox[1] < 2) tmpbbox[1] = 2;

          tempCtx.clearRect(tmpbbox[0] - 2, tmpbbox[1] - 2,
              tmpbbox[2] + 4 - tmpbbox[0], tmpbbox[3] + 4 - tmpbbox[1]);

          if (polyLabeling.resize === VERTEX) {
            if (!IfGenarate) {
              polyLabeling.redraw(minPoly);
              IfGenarate = true;
            }
            minPoly.p[minPoint] = [mouseX, mouseY];
            minPoly.changeHiddenVertex(minPoint);
            minPoly.changeHiddenVertex(
                (minPoint - 1 + minPoly.num) % minPoly.num);

            if (state === 'resize') {
              minPoly.drawPoly(minPoly.num, tempCtx);
            } else if (state === 'select_resize') {
              minPoly.drawPoly(minPoly.num, tempCtx, true);
            }
          } else if (polyLabeling.resize === MID_VERTEX) {
            if (!IfGenarate) {
              minPoly.genarateNewVertex(minMidPoint);
              polyLabeling.redraw(minPoly);
              IfGenarate = true;
            }
            minPoly.p[minMidPoint + 1] = [mouseX, mouseY];

            minPoly.changeHiddenVertex(minMidPoint);
            minPoly.changeHiddenVertex(
                (minMidPoint + 1) % minPoly.num);
            if (state === 'resize') {
              minPoly.drawPoly(minPoly.num, tempCtx);
            } else if (state === 'select_resize') {
              minPoly.drawPoly(minPoly.num, tempCtx, true);
            }
          }
        }
      }

      /**
       * Summary: To be completed.
       * @param {type} ignoredEvent: Description.
       */
      function mouseUp(ignoredEvent) {
        if (state === 'resize' ||
            state === 'select_resize') {
          IfGenarate = false;
          cnt = 0;
          if (minPoly && minPoly !== -1) {
            polylist[minPoly.listIndex] = minPoly;
          }

          if (state === 'resize') {
            tempCtx.clearRect(0, 0,
                tempCanvas.width, tempCanvas.height);
            if (!getInVertex) {
              if (minPoly.outsideList.length > 0) {
                polyLabeling.redraw();
              } else {
                minPoly.drawPoly(minPoly.num, ctx);
              }
            }
            polyLabeling.clearGlobals();
            polyLabeling.hidden_redraw();
          } else {
            tempCtx.clearRect(0, 0, tempCanvas.width,
                tempCanvas.height);
            state = 'select';
            if (!getInVertex) {
              if (minPoly.outsideList.length > 0) {
                polyLabeling.redraw();
              } else {
                minPoly.drawPoly(minPoly.num, ctx, true);
              }
            }
            polyLabeling.hidden_redraw();
          }

          minPoly.recomputeBbox();
          if (labelShowtime) {
            let pos = calcuCenter(minPoly);
            minPoly.fillLabel(pos, ctx);
          }

          let tmp;
          if (ratio) {
            tmp = [];
            for (let i = 0; i < minPoly.num; i++) {
              tmp[i] =
                  [
                    parseFloat((minPoly.p[i][0] / ratio).toFixed(6)),
                    parseFloat((minPoly.p[i][1] / ratio).toFixed(6))];
            }
          }
          addEvent('resize', minPoly.id, {
            p: tmp,
            num: minPoly.num,
            BezierOffset: minPoly.BezierOffset,
            beziernum: minPoly.beziernum,
          });
        } else if (state === 'quick_draw') {
          if (polyLabeling.resize === VERTEX) {
            if (selectCount === 0) {
              selectCount++;
              borderPoly = minPoly;
              borderPoint[0] = minPoint;
              pointList[0] = minPoly.p[minPoint];
              $('#toolhead').css('background-color', '#00C000');
            } else if (selectCount === 1) {
              borderPoint[1] = -1;
              // console.log(borderPoly, minPoly);
              if (borderPoly !== minPoly) {
                for (let i = 0; i < borderPoly.num; i++) {
                  if (borderPoly.p[i][0] ===
                      minPoly.p[minPoint][0]
                      && borderPoly.p[i][1] ===
                      minPoly.p[minPoint][1]) {
                    borderPoint[1] = i;
                    break;
                  }
                }
                if (borderPoint[1] === -1) {
                  for (let i = 0; i < minPoly.num; i++) {
                    if (minPoly.p[i][0] ===
                        borderPoly.p[borderPoint[0]][0]
                        && minPoly.p[i][1] ===
                        borderPoly.p[borderPoint[0]][1]) {
                      borderPoly = minPoly;
                      borderPoint[0] = i;
                      borderPoint[1] = minPoint;
                      break;
                    }
                  }
                }
              } else if (minPoint !== borderPoint[0]) {
                borderPoint[1] = minPoint;
              }
              if (borderPoint[1] > -1) {
                selectCount++;// find border successfully
                let start = borderPoint[0];
                let end = borderPoint[1];
                for (let i = start; i !== end;
                     i = (i + 1) % borderPoly.num) {
                  pointList1.push(borderPoly.p[i]);
                }
                pointList1.push(borderPoly.p[end]);

                for (let i = start; i !== end; i =
                    (i - 1 + borderPoly.num) % borderPoly.num) {
                  pointList2.push(borderPoly.p[i]);
                }
                pointList2.push(borderPoly.p[end]);

                pointList = pointList1;
                borderPoly = -1;
                $('#toolhead').css('background-color',
                    '#00F0E0');
              } else {
                // not on the same poly, continue to find border
                if (pointList.length > 0) {
                  for (let i = 0; i < pointList.length; i++) {
                    poly.update(pointList[i][0],
                        pointList[i][1], cnt);
                    cnt++;
                    poly.num++;
                  }
                }
                clearBorderInfo();
                selectCount++;
                borderPoly = minPoly;
                borderPoint[0] = minPoint;
                pointList[0] = minPoly.p[minPoint];
              }
            } else if (selectCount === 2) {
              if (pointList.length > 0) {
                for (let i = 0; i < pointList.length; i++) {
                  poly.update(pointList[i][0],
                      pointList[i][1], cnt);
                  cnt++;
                  poly.num++;
                }
              }// increase the founded border
              clearBorderInfo();
              selectCount++;// start find new border
              borderPoly = minPoly;
              borderPoint[0] = minPoint;
              pointList[0] = minPoly.p[minPoint];
              $('#toolhead').css('background-color', '#00C000');
            }
            if (borderPoly !== -1) {
              borderPoly.drawPoly(borderPoly.num, tempCtx, true);
            }
          }
        }
      }

      /**
       * Summary: To be completed.
       * @param {type} e: Description.
       */
      function KeyDown(e) {
        let keyID = e.KeyCode ? e.KeyCode : e.which;
        if (keyID === 27) { // Esc
          if (state === 'draw' || state === 'quick_draw') {
            cnt = 0;
            tempCtx.clearRect(0, 0, tempCanvas.width,
                tempCanvas.height);
            if (state == 'quick_draw') {
              clearBorderInfo();
              ShowMid = true;
              polyLabeling.hidden_redraw();
            }
            state = 'free';
            window.addEventListener('dblclick', DoubleClick, false);
            $('#toolhead').css('background-color', '#DDDDDD');
            // Variables should not be deleted
            // To be fixed
            poly = [];
          } else if (getInVertex === true &&
              polyLabeling.resize === VERTEX) {
            minPoly.clearControlPoints(minPoly.num, tempCtx);
            if (minPoly.degree[minPoint] === 0) {
              minPoly.deleteVertex(minPoint);
            } else {
              minPoly.deleteBezier(minPoint);
            }

            cnt = 0;
            if (minPoly.num > 1) {
              polylist[minPoly.listIndex] = minPoly;
            }

            if (minPoly.num <= 1) {
              if (targetPoly === minPoly) {
                targetPoly = -1;
                state = 'free';
              }
            }
            if (state !== 'select') {
              polyLabeling.clearGlobals();
            } else {
              getInVertex = false;
              getInPoly = false;
              polyLabeling.resize = 0;
            }
            polyLabeling.redraw();
            polyLabeling.hidden_redraw();
          } else {
            alert('Nothing to delete!');
          }
        } else if (keyID === 66) {// B or b
          if (getInVertex === true
              && polyLabeling.resize === MID_VERTEX) {
            minPoly.clearControlPoints(minPoly.num, tempCtx);
            minPoly.addBezier(minMidPoint);
            cnt = 0;

            if (state !== 'select') {
              polyLabeling.clearGlobals();
            } else {
              getInVertex = false;
              getInPoly = false;
              polyLabeling.resize = 0;
            }

            polylist[minPoly.listIndex] = minPoly;
            polyLabeling.hidden_redraw();
          } else {
            alert('Can not add Bezier Curve here!');
          }
        } else if (keyID === 77) {// M or m
          if (ShowMid) {
            ShowMid = false;
            alert('Now the midpoints are invisiable, press <H> or' +
                ' <h> again to show them. You can not add points ' +
                'and curve until you press <H> or <h> again.');
            polyLabeling.hidden_redraw();
          } else {
            ShowMid = true;
            alert('Now the midpoints are visiable, ' +
                'you can add points and curve freely. ' +
                'press <H> or <h> again to hide them.');
            polyLabeling.hidden_redraw();
          }
        } else if (keyID === 46 || keyID === 8) { // Delete or Backspace
          if (state !== 'draw' && state != 'quick_draw' &&
              getInVertex === true &&
              polyLabeling.resize === VERTEX) {
            minPoly.clearControlPoints(minPoly.num, tempCtx);

            if (minPoly.degree[minPoint] === 0) {
              minPoly.deleteVertex(minPoint);
            } else {
              minPoly.deleteBezier(minPoint);
            }
            cnt = 0;
            if (minPoly.num > 1) {
              polylist[minPoly.listIndex] = minPoly;
            }

            if (minPoly.num <= 1) {
              if (minPoly === targetPoly) {
                targetPoly = -1;
                state = 'free';
              }
            }
            if (state !== 'select') {
              polyLabeling.clearGlobals();
            } else {
              getInVertex = false;
              getInPoly = false;
              polyLabeling.resize = 0;
            }

            polyLabeling.redraw();
            polyLabeling.hidden_redraw();
          } else if (state === 'select') {
            if (targetPoly !== -1) {
              targetPoly.clearControlPoints(targetPoly.num,
                  tempCtx);
              deletePolyfromPolylist(targetPoly);
              polyLabeling.clearGlobals();
              polyLabeling.redraw();
              polyLabeling.hidden_redraw();
              cnt = 0;
            } else {
              alert(
                  'Please select the object you want to delete.');
            }
          } else if (state === 'draw') {
            if (cnt > 0) {
              poly.p.splice(cnt - 1, 1);
              poly.degree.splice(cnt - 1, 1);
              if (cnt > 1) ;
              poly.hidden_p.splice(cnt - 2, 1);
              cnt--;
              poly.num--;
              tempCtx.clearRect(0, 0, tempCanvas.width,
                  tempCanvas.height);
              if (cnt === 0) {
                state = 'free';
              } else {
                poly.drawPoly(poly.num, tempCtx);
              }
            }
            window.addEventListener('dblclick', DoubleClick, false);
          } else if (state === 'quick_draw') {
            tempCtx.clearRect(0, 0, tempCanvas.width,
                tempCanvas.height);
            if (pointList.length > 0) {
              clearBorderInfo();
              $('#toolhead').css('background-color', '#00F0E0');
              poly.drawPoly(poly.num, tempCtx, false, true,
                  false);
            } else {
              if (cnt > 0) {
                poly.p.splice(cnt - 1, 1);
                poly.degree.splice(cnt - 1, 1);
                if (cnt > 1) ;
                poly.hidden_p.splice(cnt - 2, 1);
                cnt--;
                poly.num--;
                if (cnt === 0) {
                  state = 'free';
                  ShowMid = true;
                  polyLabeling.hidden_redraw();
                  $('#toolhead').css('background-color',
                      '#DDDDDD');
                } else {
                  poly.drawPoly(poly.num, tempCtx,
                      false, true, false);
                }
              }
            }
            window.addEventListener('dblclick', DoubleClick, false);
          } else {
            alert('Nothing to delete!');
          }
        } else if (keyID === 76) { // L or l
          if (state === 'free' || state === 'select') {
            if (labelShowtime === 0) {
              labelShowtime = 1;
              polyLabeling.redraw();
              $('#label_btn').text('Hide Label (L)');
              if (state === 'select') {
                getInVertex = false;
                getInPoly = false;
                polyLabeling.resize = 0;
              }
            } else {
              labelShowtime = 0;
              polyLabeling.redraw();
              $('#label_btn').text('Show Label (L)');
              if (state === 'select') {
                getInVertex = false;
                getInPoly = false;
                polyLabeling.resize = 0;
              }
            }
          }
        } else if (keyID === 38) {// up
          if (canvasScale === 4) return;
          incHandler();
        } else if (keyID === 40) {// down
          if (canvasScale === 1) return;
          decHandler();
        } else if (keyID === 83) {// S or s
          if (state === 'draw' || state === 'free'
              || state === 'select') {
            $('#toolhead').css('background-color', '#00F0E0');
            if (state === 'draw') {
              poly.p.splice(cnt, 1);
            }
            if (state === 'free' || state === 'select') {
              targetPoly = -1;
              let catIdx = document.
                  getElementById('category_select').selectedIndex;
              let nameIdx = document.
                  getElementById('name_select').selectedIndex;
              let cate = LabelChart[catIdx][nameIdx];
              poly = new Poly(cate,
                  MaxID);
              poly.listIndex = polylist.length;
              cnt = 0;
            }
            state = 'quick_draw';
            ShowMid = false;
            polyLabeling.redraw();
            polyLabeling.hidden_redraw();
          } else if (state === 'quick_draw') {
            if (pointList.length > 0) {
              for (let i = 0; i < pointList.length; i++) {
                poly.update(pointList[i][0],
                    pointList[i][1], cnt);
                cnt++;
                poly.num++;
              }
            }
            clearBorderInfo();
            if (cnt === 0) {
              state = 'free';
              poly = [];
            } else state = 'draw';
            ShowMid = true;
            polyLabeling.hidden_redraw();
            $('#toolhead').css('background-color', '#DDDDDD');
          }
        } else if (keyID === 37 || keyID === 39) { // Left/Right Arrow
          if (state === 'quick_draw') {
            if (pointList === pointList1) {
              pointList = pointList2;
            } else {
              pointList = pointList1;
            }

            if (pointList.length > 0) {
              tempCtx.clearRect(0, 0, tempCanvas.width,
                  tempCanvas.height);
              drawLine(pointList, cnt, poly, tempCtx);
            }
          }
        } else if (keyID === 72) { // H or h
          if (showall === false) {
            showall = true;
            $('#show_all').text('Hide All (H)');
            tempCtx.clearRect(0, 0, tempCanvas.width,
                tempCanvas.height);
            polyLabeling.redraw();
          } else {
            showall = false;
            $('#show_all').text('Show All (H)');
            tempCtx.clearRect(0, 0, tempCanvas.width,
                tempCanvas.height);
            polyLabeling.redraw();
          }
        }
      }

      /**
       * Summary: To be completed.
       * @param {type} e: Description.
       */
      function DoubleClick(e) {
        if (state === 'link') return;
        let pos = computePosition(e);
        let mouseX = pos[0];
        let mouseY = pos[1];
        if (pos[0] < 0 || pos[1] < 0) {
          if (state === 'select') {
            polyLabeling.clearGlobals();
            polyLabeling.redraw();
          }
          return;
        }
        // avoid user hit more than twice
        window.removeEventListener('mousedown', mouseDown, false);
        selectedPoly = polyLabeling.selectPoly(mouseX, mouseY);
        clearTimeout(interval);
        clearTimeout(enable);

        if (state === 'draw') {
          if (cnt > 1 && poly
              && calcuDis(poly.p[cnt - 2], pos) > MIN_DIS) {
            window.addEventListener('mousedown', mouseDown, false);
            return;
          }
          cnt = 0;
          state = 'free';
          tempCtx.clearRect(0, 0, tempCanvas.width,
              tempCanvas.height);
        }
        if (selectedPoly !== -1) {
          state = 'select';
          cnt = 0;

          if (selectedPoly !== targetPoly) {
            targetPoly = selectedPoly;

            let tmp;
            let idx;
            polyLabeling.redraw();
            for (idx = 0; idx < LabelChart.length; idx++) {
              tmp = LabelChart[idx];
              if (tmp.indexOf(selectedPoly.category) !== -1) {
                $('#category_select').val(
                    LabelList[idx]);
                break;
              }
            }
            $('select#name_select').html('');
            for (let j = 0; j < LabelChart[idx].length; j++) {
              $('select#name_select').append('<option>' +
                  LabelChart[idx][j] + '</option>');
            }
            document.getElementById('name_select').setAttribute(
                'size', LabelChart[idx].length);
            $('#name_select').val(selectedPoly.category);
          }
        } else {
          if (state === 'select') {
            polyLabeling.clearGlobals();
            polyLabeling.redraw();
          }
        }

        enable = setTimeout(function() {
          window.addEventListener('mousedown', mouseDown, false);
        }, 350);
      }

      window.addEventListener('keydown', KeyDown, true);
      window.addEventListener('mousedown', mouseDown, false);
      window.addEventListener('mousemove', mouseMove, false);
      window.addEventListener('mouseup', mouseUp, false);
      window.addEventListener('dblclick', DoubleClick, false);

      $('#toolbox').mouseover(function() {
        window.removeEventListener('mousedown', mouseDown, false);
        window.removeEventListener('mousemove', mouseMove, false);
        window.removeEventListener('mouseup', mouseUp, false);
        window.removeEventListener('dblclick', DoubleClick, false);
      });
      $('#toolbox').mouseleave(function() {
        window.addEventListener('mousedown', mouseDown, false);
        window.addEventListener('mousemove', mouseMove, false);
        window.addEventListener('mouseup', mouseUp, false);
        window.addEventListener('dblclick', DoubleClick, false);
      });

      $('#control_panel').mouseover(function() {
        window.removeEventListener('mousedown', mouseDown, false);
        window.removeEventListener('mousemove', mouseMove, false);
        window.removeEventListener('mouseup', mouseUp, false);
        window.removeEventListener('dblclick', DoubleClick, false);
      });
      $('#control_panel').mouseleave(function() {
        window.addEventListener('mousedown', mouseDown, false);
        window.addEventListener('mousemove', mouseMove, false);
        window.addEventListener('mouseup', mouseUp, false);
        window.addEventListener('dblclick', DoubleClick, false);
      });
      $('#header').mouseover(function() {
        window.removeEventListener('mousedown', mouseDown, false);
        window.removeEventListener('mousemove', mouseMove, false);
        window.removeEventListener('mouseup', mouseUp, false);
        window.removeEventListener('dblclick', DoubleClick, false);
      });
      $('#header').mouseleave(function() {
        window.addEventListener('mousedown', mouseDown, false);
        window.addEventListener('mousemove', mouseMove, false);
        window.addEventListener('mouseup', mouseUp, false);
        window.addEventListener('dblclick', DoubleClick, false);
      });
      $('#temp_canvas').mouseover(function() {
        window.addEventListener('mousedown', mouseDown, false);
        window.addEventListener('mousemove', mouseMove, false);
        window.addEventListener('mouseup', mouseUp, false);
        window.addEventListener('dblclick', DoubleClick, false);
      });
      $('#hidden_canvas').mouseover(function() {
        window.addEventListener('mousedown', mouseDown, false);
        window.addEventListener('mousemove', mouseMove, false);
        window.addEventListener('mouseup', mouseUp, false);
        window.addEventListener('dblclick', DoubleClick, false);
      });
      $('#main_canvas').mouseover(function() {
        window.addEventListener('mousedown', mouseDown, false);
        window.addEventListener('mousemove', mouseMove, false);
        window.addEventListener('mouseup', mouseUp, false);
        window.addEventListener('dblclick', DoubleClick, false);
      });

      $('#delete_btn').click(function() {
        // ctx.putImageData(lastScene,0,0);
        if (targetPoly != -1) {
          deletePolyfromPolylist(targetPoly);
          polyLabeling.clearGlobals();
          polyLabeling.redraw();
          polyLabeling.hidden_redraw();
          cnt = 0;
        } else {
          alert('Please select the object you want to delete.');
        }
      });
      $('#show_all').click(function() {
        if (showall === false) {
          showall = true;
          $('#show_all').text('Hide All (H)');
          tempCtx.clearRect(0, 0, tempCanvas.width,
              tempCanvas.height);
          polyLabeling.redraw();
        } else {
          showall = false;
          $('#show_all').text('Show All (H)');
          tempCtx.clearRect(0, 0, tempCanvas.width,
              tempCanvas.height);
          polyLabeling.redraw();
        }
      });
      $('#increase_btn').click(function() {
        incHandler();
      });

      $('#decrease_btn').click(function() {
        decHandler();
      });

      $('#inside_btn').click(function() {
        if (targetPoly == -1) {
          alert('Please select a polygon that you want to draw on!');
        } else if (drawOn == false) {
          drawOn = true;
          state = 'free';
          $('#inside_btn').text('Finish');
        } else {
          drawOn = false;
          targetPoly = -1;
          state = 'free';
          polyLabeling.redraw();
          $('#inside_btn').text('On Top');
        }
      });

      $('#link_btn').click(function() {
        if (state === 'free') {
          state = 'link';
          $('#link_btn').text('Finish Link');
        } else if (state === 'link') {
          polyLabeling.linkPolyHandler();
          polyLabeling.clearGlobals();
          polyLabeling.redraw();
          $('#link_btn').text('Link');
        } else if (state == 'select') {
          if (targetPoly !== -1) {
            linkList.push(targetPoly);
            targetPoly = -1;
            state = 'link';
            $('#link_btn').text('Finish Link');
          }
        }
      });
      $('#label_btn').click(function() {
        if (state == 'free' || state == 'select') {
          if (labelShowtime === 0) {
            labelShowtime = 1;
            polyLabeling.redraw();
            $('#label_btn').text('Hide Label (L)');
            if (state === 'select') {
              getInVertex = false;
              getInPoly = false;
              polyLabeling.resize = 0;
            }
          } else {
            labelShowtime = 0;
            polyLabeling.redraw();
            $('#label_btn').text('Show Label (L)');
            if (state === 'select') {
              getInVertex = false;
              getInPoly = false;
              polyLabeling.resize = 0;
            }
          }
        }
      });

      $('#category_select').change(function() {
        let catIdx =
            document.getElementById('category_select').selectedIndex;
        document.getElementById('name_select').setAttribute(
            'size', LabelChart[catIdx].length);
        $('select#name_select').html('');
        for (let j = 0; j < LabelChart[catIdx].length; j++) {
          $('select#name_select').append('<option>'
              + LabelChart[catIdx][j] + '</option>');
        }
        $('#name_select').val(LabelChart[catIdx][0]);

        if (state == 'select') {
          let nameIdx =
              document.getElementById('name_select').selectedIndex;
          let cate = LabelChart[catIdx][nameIdx];
          cnt = 0;
          if (targetPoly !== -1) {
            if (targetPoly.sameIdList.length === 0) {
              polylist[targetPoly.listIndex].category = cate;
            } else {
              for (let key in targetPoly.sameIdList) {
                if (polylist[targetPoly.sameIdList[key]].
                        hasOwnProperty('category')) {
                  polylist[targetPoly.sameIdList[key]].
                      category = cate;
                }
              }
            }
            let pos = calcuCenter(targetPoly);
            polyLabeling.redraw();
            targetPoly.fillLabel(pos, ctx);
          }
          getInVertex = false;
          getInPoly = false;
          polyLabeling.resize = 0;
          // $("#clear_btn").attr("disabled", false);
        }
      });

      $('#name_select').change(function() {
        let catIdx =
            document.getElementById('category_select').selectedIndex;
        if (state == 'select') {
          let nameIdx =
              document.getElementById('name_select').selectedIndex;
          let cate = LabelChart[catIdx][nameIdx];
          cnt = 0;
          if (targetPoly !== -1) {
            if (targetPoly.sameIdList.length === 0) {
              polylist[targetPoly.listIndex].category = cate;
            } else {
              for (let key in targetPoly.sameIdList) {
                if (polylist[targetPoly.sameIdList[key]].
                        hasOwnProperty('category')) {
                  polylist[targetPoly.sameIdList[key]].
                      category = cate;
                }
              }
            }
            let pos = calcuCenter(targetPoly);
            polyLabeling.redraw();
            targetPoly.fillLabel(pos, ctx);
          }
          getInVertex = false;
          getInPoly = false;
          polyLabeling.resize = 0;
          // $("#clear_btn").attr("disabled", false);
        }
      });
    };

    return PolyLabeling;
  })();

  let Poly;

  Poly = (function() {
    /**
     * Summary: To be completed.
     * @param {type} fixedLabel: category.
     * @param {type} id: id.
     * sameIdList: list_indexs of polys that are linked as the same
     *             object using LINK.
     * insideList: list_indexs of polys that are labeled inside of
     *             this label using ON TOP.
     * outsideList: list_indexs of polys this poly was labeld on top of
     */
    function Poly(fixedLabel, id) {
      SatLabel.call(this, null, id);
      this.id = id;
      this.listIndex = 0;
      this.num = 0;
      this.beziernum = 0;
      this.p = [];
      this.degree = [];
      this.BezierOffset = [];
      this.hidden_p = [];
      this.category = fixedLabel;
      this.sameIdList = [];
      this.insideList = [];
      this.outsideList = [];
      this.bbox = [10000, 10000, -1, -1];
    }

    Poly.prototype = Object.create(SatLabel.prototype);

    Poly.prototype.update = function(clickX, clickY, cnt) {
      let vec = [clickX, clickY];
      let prevX;
      let prevY;
      this.p[cnt] = vec;
      this.degree[cnt] = 0;

      if (clickX < this.bbox[0]) this.bbox[0] = clickX;
      if (clickX > this.bbox[2]) this.bbox[2] = clickX;
      if (clickY < this.bbox[1]) this.bbox[1] = clickY;
      if (clickY > this.bbox[3]) this.bbox[3] = clickY;

      if (cnt > 0) {
        prevX = this.p[cnt - 1][0];
        prevY = this.p[cnt - 1][1];
        this.hidden_p[cnt - 1] = [
          (clickX + prevX) / 2,
          (clickY + prevY) / 2];
      }

      let tmp;
      if (ratio) {
        tmp = [
          parseFloat((clickX / ratio).toFixed(6)),
          parseFloat((clickY / ratio).toFixed(6))];
      }
      addEvent('drawPoly', this.id, {
        p: tmp,
        num: cnt,
      });
      return true;
    };

    Poly.prototype.computeDegree = function() {
      // degree: 1 if end points of B curve;
      // 2 if mid points of B curve; 0 otherwise
      for (let i = 0; i < this.num; i++) {
        this.degree[i] = 0;
      }
      if (this.beziernum > 0) {
        for (let j = 0; j < this.beziernum; j++) {
          let start = this.BezierOffset[j];
          this.degree[start]++;
          this.degree[start + 1] = 2;
          this.degree[start + 2] = 2;
          this.degree[(start + 3) % this.num]++;
        }
      }
    };

    Poly.prototype.addBezier = function(index) {
      let prev = this.p[index];
      let next = this.p[(index + 1) % this.num];

      this.num += 2;
      let mid1 = [
        Math.round((prev[0] * 2 +
            next[0]) / 3), Math.round((prev[1] * 2 +
            next[1]) / 3)];
      let mid2 = [
        Math.round((prev[0] +
            next[0] * 2) / 3), Math.round((prev[1] +
            next[1] * 2) / 3)];

      this.p.splice(index + 1, 0, mid1, mid2);

      this.hidden_p.splice(index, 1);
      this.hidden_p.splice(index, 0, 0, 0, 0);

      this.changeHiddenVertex(index);
      this.changeHiddenVertex(index + 1);
      this.changeHiddenVertex(index + 2);

      this.degree.splice(index + 1, 0, 2, 2);
      this.degree[index]++;

      this.degree[(index + 3) % this.num]++;
      // console.log(this.degree);
      this.beziernum++;

      this.BezierOffset.push(index);
      this.BezierOffset.sort(compare);
      for (let i = 0; i < this.beziernum; i++) {
        if (this.BezierOffset[i] > index) {
          this.BezierOffset[i] += 2;
        }
      }
      addEvent('addBezier', this.id, {
        bezieroffset: this.BezierOffset,
        beziernum: this.beziernum,
      });
    };
    Poly.prototype.deleteBezier = function(index) {
      let offset = -1;
      for (let i = 0; i < this.beziernum; i++) {
        if (this.BezierOffset[i] <= index &&
            this.BezierOffset[i] + 3 >= index) {
          offset = i;
          break;
        } else if (index === 0 && this.BezierOffset[i] + 3
            === this.num) {
          offset = i;
          break;
        }
      }
      if (offset === -1) {
        alert('cannot delete bezier');
        return;
      }

      let start = this.BezierOffset[offset];

      this.num -= 2;
      this.beziernum--;

      this.BezierOffset.splice(offset, 1);
      for (let i = offset; i < this.beziernum; i++) {
        this.BezierOffset[i] -= 2;
      }
      this.p.splice(start + 1, 2);
      this.degree.splice(start + 1, 2);
      this.degree[start]--;
      this.degree[(start + 1) % this.num]--;
      this.hidden_p.splice(start, 3);

      let hid = [
        (this.p[start][0] +
            this.p[(start + 1) % this.num][0]) / 2,
        (this.p[start][1] +
            this.p[(start + 1) % this.num][1]) / 2];

      this.hidden_p.splice(start, 0, hid);

      addEvent('delBezier', this.id, {
        bezieroffset: this.BezierOffset,
        beziernum: this.beziernum,
      });
    };

    Poly.prototype.drawPoly = function(cnt, context, select,
                                       state, vacant) {
      let width = 1;
      let withBezier = false;
      // context.globalCompositeOperation="destination-out";
      if (this.beziernum > 0) withBezier = true;
      if (withBezier === false) {
        this.drawPolyWithoutBezier(cnt, context);
      } else if (withBezier === true && cnt > 4) {
        this.drawPolyWithBezier(cnt, context);
      }
      context.closePath();

      if (state) {
        width = 2;
      } else if (select) {
        context.fillStyle = SELECT_COLOR;
        context.fill();
      } else if (vacant) {
        context.save();
        context.globalCompositeOperation = 'destination-out';
        context.fillStyle = this.styleColor(1);
        context.fill();
      } else {
        context.fillStyle = this.styleColor(ALPHA);
        context.fill();
      }

      context.lineWidth = width / canvasScale;

      context.strokeStyle = this.styleColor(1);
      context.stroke();

      if (withBezier && cnt === 4) {
        let index = this.BezierOffset[0];
        context.beginPath();
        context.moveTo(this.p[index][0], this.p[index][1]);
        context.bezierCurveTo(this.p[(index + 1) % cnt][0],
            this.p[(index + 1) % cnt][1],
            this.p[(index + 2) % cnt][0], this.p[(index + 2) % cnt][1],
            this.p[(index + 3) % cnt][0], this.p[(index + 3) % cnt][1]);
      }
      // TODO(Fisher): delete this if not useful
      // if (context === hiddenCtx) {
      //   context.strokeStyle = this.colors(this.listIndex, 1);
      // } else {
      //   context.strokeStyle = this.styleColor(1);
      // }
      context.strokeStyle = this.styleColor(1);

      context.stroke();

      if (withBezier) {
        this.drawDashLine(cnt, context);
      }
      if (vacant) {
        context.restore();
      }

      if (this.insideList.length > 0) {
        for (let key in this.insideList) {
          if (polylist[this.insideList[key]].hasOwnProperty('num')) {
            polylist[this.insideList[key]].drawPoly(
                polylist[this.insideList[key]].num, context,
                false, false, true);
          }
        }

        for (let key in this.insideList) {
          if (polylist[this.insideList[key]].hasOwnProperty('num')) {
            polylist[this.insideList[key]].drawPoly(
                polylist[this.insideList[key]].num, context,
                false, false, false);
          }
        }
      }
    };

    Poly.prototype.drawPolyWithoutBezier = function(cnt, context) {
      context.beginPath();
      context.moveTo(this.p[0][0], this.p[0][1]);
      for (let j = 1; j < cnt; j++) {
        context.lineTo(this.p[j][0], this.p[j][1]);
      }
      context.closePath();
    };

    Poly.prototype.drawPolyWithBezier = function(cnt, context) {
      let num = this.beziernum;
      if (num === 0) return;
      let index = this.BezierOffset[0];
      let nextIndex;

      context.beginPath();
      context.moveTo(this.p[index][0], this.p[index][1]);

      for (let i = 0; i < num - 1; i++) {
        index = this.BezierOffset[i];
        nextIndex = this.BezierOffset[i + 1];

        context.bezierCurveTo(this.p[(index + 1) % cnt][0],
            this.p[(index + 1) % cnt][1],
            this.p[(index + 2) % cnt][0], this.p[(index + 2) % cnt][1],
            this.p[(index + 3) % cnt][0], this.p[(index + 3) % cnt][1]);
        if (nextIndex !== (index + 4) % cnt &&
            nextIndex !== (index + 3) % cnt) {
          for (let j = (index + 4) % cnt;
               j !== nextIndex; j = (j + 1) % cnt) {
            context.lineTo(this.p[j][0], this.p[j][1]);
          }
          context.lineTo(this.p[nextIndex][0], this.p[nextIndex][1]);
        } else if (nextIndex === (index + 4) % cnt) {
          context.lineTo(this.p[nextIndex][0], this.p[nextIndex][1]);
        }
      }

      nextIndex = this.BezierOffset[this.beziernum - 1];
      index = this.BezierOffset[0];

      context.bezierCurveTo(this.p[(nextIndex + 1) % cnt][0],
          this.p[(nextIndex + 1) % cnt][1],
          this.p[(nextIndex + 2) % cnt][0],
          this.p[(nextIndex + 2) % cnt][1],
          this.p[(nextIndex + 3) % cnt][0],
          this.p[(nextIndex + 3) % cnt][1]);

      if (index !== (nextIndex + 4) % cnt
          && index !== (nextIndex + 3) % cnt) {
        for (let j = (nextIndex + 4) % cnt;
             j !== index; j = (j + 1) % cnt) {
          context.lineTo(this.p[j][0], this.p[j][1]);
        }
        context.lineTo(this.p[index][0], this.p[index][1]);
      } else if (index === (nextIndex + 4) % cnt) {
        context.lineTo(this.p[index][0], this.p[index][1]);
      }
    };

    Poly.prototype.drawDashLine = function(cnt, context) {
      let num = this.beziernum;
      if (num === 0) return;
      let index = this.BezierOffset[0];
      context.setLineDash([3]);

      for (let i = 0; i < num; i++) {
        index = this.BezierOffset[i];
        // nextIndex is not defined
        // nextIndex = this.BezierOffset[(i + 1) % num];
        context.moveTo(this.p[index][0], this.p[index][1]);
        context.lineTo(this.p[(index + 1) % cnt][0],
            this.p[(index + 1) % cnt][1]);
        context.lineTo(this.p[(index + 2) % cnt][0],
            this.p[(index + 2) % cnt][1]);
        context.lineTo(this.p[(index + 3) % cnt][0],
            this.p[(index + 3) % cnt][1]);

        context.stroke();
      }
      context.setLineDash([]);
    };

    Poly.prototype.fillLabel = function(pos, context) {
      let word = this.category.substring(0, 3);
      let tagWidth = word.length;
      let tagHeight = 15;
      context.font = '13px Verdana';
      if (canvasScale === 4) {
        context.font = '9px Verdana';
        tagHeight = 10;
        tagWidth = word.length - 1;
      }
      context.fillStyle = this.styleColor(1);
      context.fillRect(pos[0] - tagWidth, pos[1]
          - 2, tagWidth * 9, tagHeight);
      context.fillStyle = 'rgb(0, 0, 0)';
      context.fillText(word, pos[0] - tagWidth, pos[1] + tagHeight
          - 3);
    };

    // draw poly on hidden canvas
    Poly.prototype.drawHiddenPoly = function(cnt) {
      hiddenCtx.fillStyle = this.hidden_colors(this.listIndex, -1);
      hiddenCtx.strokeStyle = this.hidden_colors(this.listIndex, -1);

      let withBezier = false;
      if (this.beziernum > 0) withBezier = true;
      if (withBezier === false) {
        this.drawPolyWithoutBezier(cnt, hiddenCtx);
      } else if (withBezier === true && cnt > 4) {
        this.drawPolyWithBezier(cnt, hiddenCtx);
      }
      hiddenCtx.closePath();
      hiddenCtx.fill();
      hiddenCtx.lineWidth = 2 / canvasScale;
      hiddenCtx.stroke();

      if (withBezier && cnt === 4) {
        let index = this.BezierOffset[0];
        hiddenCtx.beginPath();
        hiddenCtx.moveTo(this.p[index][0], this.p[index][1]);
        hiddenCtx.bezierCurveTo(this.p[(index + 1) % cnt][0],
            this.p[(index + 1) % cnt][1],
            this.p[(index + 2) % cnt][0], this.p[(index + 2) % cnt][1],
            this.p[(index + 3) % cnt][0], this.p[(index + 3) % cnt][1]);
      }
      hiddenCtx.stroke();
      this.drawHiddenVertex(cnt);
    };

    Poly.prototype.genarateNewVertex = function(index) {
      for (let i = this.num - 1; i > index; i--) {
        this.p[i + 1] = this.p[i];
        this.hidden_p[i + 1] = this.hidden_p[i];
      }
      this.p[index + 1] = this.hidden_p[index];
      this.num++;
      this.degree.splice(index + 1, 0, 0);
      for (let i = 0; i < this.beziernum; i++) {
        if (this.BezierOffset[i] >= index + 1) {
          this.BezierOffset[i]++;
        }
      }
      this.changeHiddenVertex(index);
      this.changeHiddenVertex(index + 1);
      addEvent('addvertex', this.id, {
        new_index: index,
      });
    };

    Poly.prototype.deleteVertex = function(index) {
      if (this.num <= this.beziernum * 3) {
        return;
      }
      this.num--;
      for (let i = 0; i < this.beziernum; i++) {
        this.BezierOffset[i] = this.BezierOffset[i] % this.num;
      }

      this.p.splice(index, 1);
      this.hidden_p.splice(index, 1);
      this.degree.splice(index, 1);
      if (this.num < 2) {
        deletePolyfromPolylist(this);
        return;
      }
      for (let i = 0; i < this.beziernum; i++) {
        if (this.BezierOffset[i] > index) {
          this.BezierOffset[i]--;
        }
      }
      let prev = (index - 1 + this.num) % this.num;

      this.changeHiddenVertex(prev);
      addEvent('delvertex', this.id, {
        del_index: index,
      });
    };

    Poly.prototype.changeHiddenVertex = function(index) {
      let next = (index + 1) % this.num;
      let newIndex = index % this.num;

      this.hidden_p[newIndex] = [
        (this.p[newIndex][0] +
            this.p[next][0]) / 2, (this.p[newIndex][1] +
            this.p[next][1]) / 2];
    };

    Poly.prototype.drawHiddenVertex = function(cnt) {
      if (this.num > 1 && ShowMid) {
        for (let i = 0; i < cnt; i++) {
          if (cnt === 2 && i === 1) break;
          if (this.degree[i] < 2 && this.degree[(i + 1) % cnt] < 2) {
            hiddenCtx.beginPath();
            if (calcuDis(this.hidden_p[i], this.p[i]) > MIN_DIS * 2
                && calcuDis(this.hidden_p[i],
                    this.p[(i + 1) % cnt]) > MIN_DIS * 2) {
              hiddenCtx.arc(this.hidden_p[i][0], this.hidden_p[i][1],
                  4 / canvasScale, 0, 2 * Math.PI, false);
            } else {
              hiddenCtx.arc(this.hidden_p[i][0], this.hidden_p[i][1],
                  3 / canvasScale, 0, 2 * Math.PI, false);
            }
            hiddenCtx.closePath();
            hiddenCtx.fillStyle =
                this.hidden_colors(this.listIndex, 2 * i + 1);
            hiddenCtx.fill();
          }
        }
      }

      for (let j = 0; j < cnt; j++) {
        hiddenCtx.beginPath();
        hiddenCtx.arc(this.p[j][0], this.p[j][1],
            4 / canvasScale, 0, 2 * Math.PI, false);
        hiddenCtx.closePath();
        hiddenCtx.fillStyle =
            this.hidden_colors(this.listIndex, 2 * j);
        hiddenCtx.fill();
      }
    };

    Poly.prototype.clearControlPoints = function(cnt, context) {
      let radius = 7 / canvasScale;
      let startpos;
      let endpos;
      for (let j = 0; j < cnt; j++) {
        startpos = [this.p[j][0] - radius - 1, this.p[j][1] - radius - 1];
        if (startpos[0] < 0) startpos[0] = 0;
        if (startpos[1] < 0) startpos[1] = 0;
        endpos = [
          this.p[j][0] + radius + 1,
          this.p[j][1] + radius + 1];
        context.clearRect(startpos[0], startpos[1],
            endpos[0], endpos[1]);
        startpos = [
          this.hidden_p[j][0] - radius - 1,
          this.hidden_p[j][1] - radius - 1];
        endpos = [
          this.hidden_p[j][0] + radius + 1,
          this.hidden_p[j][1] + radius + 1];
        context.clearRect(startpos[0], startpos[1],
            endpos[0], endpos[1]);
      }
    };

    // show points when hovering mouse on the object
    // @param cnt: # of vertices
    Poly.prototype.showControlPoints = function(cnt, context) {
      let radius = 4 / canvasScale;
      context.fillStyle = this.styleColor(0.7);
      for (let j = 0; j < cnt; j++) {
        context.beginPath();
        context.arc(this.p[j][0], this.p[j][1],
            radius, 0, 2 * Math.PI, false);
        context.closePath();
        context.fill();
      }

      context.fillStyle = MID_CONTROL_POINT;
      for (let m = 0; m < cnt; m++) {
        if (this.degree[m] < 2 && this.degree[(m + 1) % cnt] < 2
            && ShowMid) {
          context.beginPath();
          context.arc(this.hidden_p[m][0], this.hidden_p[m][1],
              radius, 0, 2 * Math.PI, false);
          context.closePath();
          context.fill();
        }
      }

      if (this.beziernum > 0) {
        context.fillStyle = BEZIER_COLOR;
        for (let i = 0; i < this.beziernum; i++) {
          for (let k = this.BezierOffset[i];
               k !== (this.BezierOffset[i] + 4) % this.num;
               k = (k + 1) % this.num) {
            context.beginPath();
            context.arc(this.p[k][0], this.p[k][1],
                radius, 0, 2 * Math.PI, false);
            context.closePath();
            context.fill();
          }
        }
      }
    };

    Poly.prototype.recomputeBbox = function() {
      this.bbox = [10000, 10000, -1, -1];
      for (let i = 0; i < this.num; i++) {
        if (this.p[i][0] < this.bbox[0]) this.bbox[0] = this.p[i][0];
        if (this.p[i][0] > this.bbox[2]) this.bbox[2] = this.p[i][0];
        if (this.p[i][1] < this.bbox[1]) this.bbox[1] = this.p[i][1];
        if (this.p[i][1] > this.bbox[3]) this.bbox[3] = this.p[i][1];
      }
    };

    Poly.prototype.hidden_colors = function(x, num) {
      let y;
      let z;
      num += 1;
      z = num & 255;
      y = (num & (255 << 8)) >> 8;
      return 'rgb(' + (x + 1) + ',' + y + ',' + z + ')';
    };
    return Poly;
  })();
}).call(this);
