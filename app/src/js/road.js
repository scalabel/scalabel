/* global SatImage SatLabel  numPoly:true ratio:true sourceImage:true
 imageList:true lastScene:true
 currentIndex:true numPoly:true assignment:true HidlastScene:true
*/

(function() {
  let polylist = [];
  let mainCanvas = document.getElementById('main_canvas');
  let ctx = mainCanvas.getContext('2d');
  let hiddenCanvas = document.getElementById('hidden_canvas');
  let hiddenCtx = hiddenCanvas.getContext('2d');
  let magnifier = document.getElementById('main_bigger');
  let magnifierCtx = magnifier.getContext('2d');
  let hiddenBigger = document.getElementById('hidden_bigger');
  let hiddenBiggerCtx = hiddenBigger.getContext('2d');
  let lastImg;
  let lastHidImg;
  let targetPoly = -1;
  let labelShowtime = 1;
  let getInVertex = false;
  let getInPoly = false;
  let state;
  let styleWidth;
  let styleHeight;
  let magnify = false;
  let ShowMid = true;

  let MIN_DIS = 6;
  let VERTEX = 1;
  let MID_VERTEX = 2;
  let MAX_DIS = 10000;
  let VERTEX_FILL = 'rgb(100,200,100)';
  let MID_VERTEX_FILL = 'rgb(200,100,100)';
  let MID_CONTROL_POINT = 'rgb(233, 133, 166)';
  let BEZIER_COLOR = 'rgb(200,200,100)';
  let SELECT_COLOR = 'rgba(100,0,100,0.5)';
  let ALPHA = 0.3;
  let MagRatio = 3;


  /**
   * Summary: To be completed.
   * @param {type} point1: Description.
   * @param {type} point2: Description.
   * @return {type} Description.
   */
  function calcuDis(point1, point2) {
    return Math.abs(point1[0] - point2[0]) +
        Math.abs(point1[1] - point2[1]);
  }

  /**
   * Summary: To be completed.
   *
   */
  function captureScene() {
    lastScene = ctx.getImageData(0, 0,
        mainCanvas.width, mainCanvas.height);
    if (magnify) {
      HidlastScene = hiddenBiggerCtx.getImageData(0, 0,
          hiddenBigger.width, hiddenBigger.height);
    }
  }

  /**
   * Summary: To be completed.
   * @param {type} v1: Description.
   * @param {type} v2: Description.
   * @return {int} Description.
   */
  function compare(v1, v2) {
    if (v1 < v2) return -1;
    else if (v1 > v2) return 1;
    else return 0;
  }

  /**
   * Summary: To be completed.
   *
   */
  function resizeCanvas() {
    ratio = parseFloat(window.innerWidth / (1.35 * sourceImage.width));
    if (parseFloat(window.innerHeight
            / (1.35 * sourceImage.height)) < ratio) {
      ratio = parseFloat(window.innerHeight
          / (1.35 * sourceImage.height));
    }
    ratio = parseFloat(ratio.toFixed(6));

    styleWidth = Math.round(sourceImage.width * ratio);
    styleHeight = Math.round(sourceImage.height * ratio);

    hiddenBigger.style.width = styleWidth + 'px';
    hiddenBigger.style.height = styleHeight + 'px';
    mainCanvas.style.width = styleWidth + 'px';
    mainCanvas.style.height = styleHeight + 'px';
    hiddenCanvas.style.width = styleWidth + 'px';
    hiddenCanvas.style.height = styleHeight + 'px';

    hiddenBigger.height = styleHeight * MagRatio;
    hiddenBigger.width = styleWidth * MagRatio;

    if (window.devicePixelRatio) {
      mainCanvas.height = styleHeight * window.devicePixelRatio;
      mainCanvas.width = styleWidth * window.devicePixelRatio;
      hiddenCanvas.height = styleHeight * window.devicePixelRatio;
      hiddenCanvas.width = styleWidth * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      hiddenCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
    } else {
      mainCanvas.height = styleHeight;
      mainCanvas.width = styleWidth;
      hiddenCanvas.height = styleHeight;
      hiddenCanvas.width = styleWidth;
    }

    hiddenBiggerCtx.scale(MagRatio, MagRatio);
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

  this.PolyLabeling = (function() {
    /**
     * Summary: To be completed.
     * @param {type} options: Description.
     * @return {type} Description.
     */
    function PolyLabeling(options) {
      SatImage.call(this, options);
      this.options = options;
      $('#main_canvas').css({
        'background-image': 'url(\'' + this.options.url + '\')',
        'cursor': 'crosshair',
      });
      $('#hidden_bigger').css({
        'background-image': 'url(\'' + this.options.url + '\')',
      });
      // Start listening to events happening in the main panel
      return this.eventController();
    }

    PolyLabeling.prototype = Object.create(SatImage.prototype);

    PolyLabeling.prototype.updateImage = function(url) {
      sourceImage = new Image();
      sourceImage.src = url;
      this.options.url = url;
      let polyLabeling = this;
      $('#main_canvas').css({
        'background-image': 'url(\'' + url + '\')',
        'cursor': 'crosshair',
      });

      $('#hidden_bigger').css({
        'background-image': 'url(\'' + url + '\')',
      });
      if (sourceImage.complete) {
        resizeCanvas();
        polyLabeling.replay();
      } else {
        sourceImage.onload = function() {
          resizeCanvas();
          polyLabeling.replay();
        };
      }
    };

    PolyLabeling.prototype.drawSelectVertex = function(mode, pos) {
      let color = VERTEX_FILL;
      if (mode === MID_VERTEX) color = MID_VERTEX_FILL;

      lastImg = ctx.getImageData(0, 0,
          mainCanvas.width, mainCanvas.height);
      ctx.beginPath();
      ctx.arc(pos[0], pos[1], 6, 0,
          2 * Math.PI, false);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();

      if (magnify) {
        lastHidImg = hiddenBiggerCtx.getImageData(0, 0,
            hiddenBigger.width, hiddenBigger.height);
        hiddenBiggerCtx.beginPath();
        hiddenBiggerCtx.arc(pos[0],
            pos[1], 6 / MagRatio, 0,
            2 * Math.PI, false);
        hiddenBiggerCtx.closePath();
        hiddenBiggerCtx.fillStyle = color;
        hiddenBiggerCtx.fill();
      }
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
              p: tmp,
              num: poly.num,
              BezierOffset: poly.BezierOffset,
              beziernum: poly.beziernum,
            },
          };
          // console.log(output);
          this.output_labels.push(output);
        }
      }
    };

    PolyLabeling.prototype.startMagnifier = function() {
      magnifier.style.display = 'inline';
      $('#shape_btn').text('Disable (M)');
      $('#main_bigger').css({
        'cursor': 'crosshair',
      });
      let poly;
      magnifierCtx.clearRect(0, 0, magnifier.width, magnifier.height);
      magnify = true;
      hiddenBiggerCtx.clearRect(0, 0, hiddenBigger.width,
          hiddenBigger.height);

      if (polylist) {
        for (let key in polylist) {
          if (polylist[key] && polylist[key] !== -1) {
            poly = polylist[key];
            poly.drawPoly(poly.num, hiddenBiggerCtx);
          }
        }
      }

      if (targetPoly !== -1 &&
          (state == 'select' || state == 'select_resize')) {
        targetPoly.drawPoly(targetPoly.num, hiddenBiggerCtx, true);
      }
      HidlastScene = hiddenBiggerCtx.getImageData(0, 0,
          hiddenBigger.width, hiddenBigger.height);
    };

    PolyLabeling.prototype.removeMagnifier = function() {
      magnify = false;
      $('#shape_btn').text('Magnify (M)');
      magnifierCtx.clearRect(0, 0, magnifier.width, magnifier.height);
      magnifier.style.display = 'none';
      magnifier.style.left = '0px';
      magnifier.style.top = '0px';
    };

    PolyLabeling.prototype.redraw = function(exception) {
      let poly;
      ctx.clearRect(0, 0,
          mainCanvas.width, mainCanvas.height);
      if (magnify) {
        hiddenBiggerCtx.clearRect(0, 0,
            hiddenBigger.width, hiddenBigger.height);
      }

      if (polylist) {
        for (let key in polylist) {
          if (exception && polylist[key] == exception) {
            continue;
          } else {
            poly = polylist[key];
            poly.drawPoly(poly.num, ctx);
            if (magnify) {
              poly.drawPoly(poly.num, hiddenBiggerCtx);
            }
          }
        }
      }

      if (targetPoly !== -1 && state === 'select') {
        targetPoly.drawPoly(targetPoly.num, ctx, true);
        if (magnify) {
          targetPoly.drawPoly(targetPoly.num,
              hiddenBiggerCtx, true);
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
      state = 'free';
      // $("#submit_btn").attr("disabled", false);
    };

    PolyLabeling.prototype.clearAll = function() {
      ctx.clearRect(0, 0,
          mainCanvas.width, mainCanvas.height);
      hiddenCtx.clearRect(0, 0,
          hiddenCanvas.width, hiddenCanvas.height);
      hiddenBiggerCtx.clearRect(0, 0,
          hiddenBigger.width, hiddenBigger.height);

      polylist = [];
      labelShowtime = 1;
      $('#label_btn').text('Hide Label (L)');
      magnify = false;
      $('#shape_btn').text('Magnify (M)');
      this.clearGlobals();
      magnifier.style.display = 'none';
      magnifier.style.left = '0px';
      magnifier.style.top = '0px';
    };

    PolyLabeling.prototype.replay = function() {
      // let polyLabeling;
      // polyLabeling = this;
      // console.log(ratio);
      if (typeof(ratio) === 'undefined' || ratio < 0) {
        alert('Error when preloading. Please refresh page.');
        return;
      }
      polylist = [];
      let poly;
      let label;
      let labelList = imageList[currentIndex].labels;

      ctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
      hiddenCtx.clearRect(0, 0,
          hiddenCanvas.width, hiddenCanvas.height);
      hiddenBiggerCtx.clearRect(0, 0,
          hiddenBigger.width, hiddenBigger.height);

      if (labelList) {
        for (let key in labelList) {
          if (!labelList.hasOwnProperty(key)) continue;
          label = labelList[key];
          poly = new Poly(label.category, parseInt(label.id));
          for (let i = 0; i < label.position.num; i++) {
            poly.p.push([
              parseFloat((label.position.p[i][0] * ratio).
                  toFixed(6)),
              parseFloat((label.position.p[i][1] * ratio).
                  toFixed(6))]);
            if (i > 0) {
              poly.hidden_p[i - 1] =
                  [
                    (poly.p[i][0] + poly.p[i - 1][0]) / 2,
                    (poly.p[i][1] + poly.p[i - 1][1]) / 2];
            }
          }
          poly.hidden_p[label.position.num - 1] =
              [
                (poly.p[label.position.num - 1][0] + poly.p[0][0]) / 2,
                (poly.p[label.position.num - 1][1] + poly.p[0][1]) / 2];

          poly.num = label.position.num;
          poly.BezierOffset = label.position.BezierOffset;
          poly.beziernum = label.position.beziernum;

          poly.computeDegree();
          poly.drawHiddenPoly(poly.num);
          poly.drawPoly(poly.num, ctx);
          polylist.push(poly);
        }
      }
    };

    PolyLabeling.prototype.selectPoly = function(clickX, clickY) {
      let r = window.devicePixelRatio;
      let pixelData;
      if (r) {
        pixelData = hiddenCtx.
            getImageData(r * clickX, r * clickY, 1, 1).data;
      } else {
        pixelData = hiddenCtx.getImageData(clickX, clickY, 1, 1).data;
      }
      let selectedPoly = -1;
      let id;
      if (pixelData[2] != 0 && pixelData[3] == 255) {
        id = pixelData[2] - 1;
        selectedPoly = polylist[id];
      } else if (pixelData[0] != 0 && pixelData[3] == 255) {
        id = pixelData[0] - 1;
        selectedPoly = polylist[id];
      }

      return selectedPoly;
    };

    PolyLabeling.prototype.selectPolyVertex = function(clickX, clickY) {
      let r = window.devicePixelRatio;
      let pixelData;
      if (r) {
        pixelData = hiddenCtx.
            getImageData(r * clickX, r * clickY, 1, 1).data;
      } else {
        pixelData = hiddenCtx.getImageData(clickX, clickY, 1, 1).data;
      }

      let selectedVertex = -1;
      if (pixelData[1] != 0 && pixelData[3] == 255) {
        selectedVertex = pixelData[1] - 1;
      }
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
      let startpos = [];
      let curpos = [];
      let MODE = 0;
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

        if (state == 'free' || state == 'select') {
          if (mouseX < 0 || mouseY < 0 ||
              mouseX > styleWidth || mouseY > styleHeight) {
            return [-1, -1];
          }
        } else {
          if (mouseX < 0) mouseX = 0.000000;
          if (mouseY < 0) mouseY = 0.000000;
          if (mouseX > styleWidth) mouseX = styleWidth;
          if (mouseY > styleHeight) mouseY = styleHeight;
        }
        return [mouseX, mouseY];
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
              && !polyLabeling.resize) {
            polyLabeling.clearGlobals();
            polyLabeling.redraw();
            cnt = 0;
            state = 'free';
          }
          if (state == 'select') {
            if (polyLabeling.resize) {
              state = 'select_resize';
            }
          } else if (cnt === 0 && state == 'free') {
            state = 'draw';
            if (!polyLabeling.resize) {
              let catIdx = document.
                  getElementById('category_select').selectedIndex;
              let cate = assignment.category[catIdx];
              poly = new Poly(cate,
                  polylist.length);

              startpos = [clickX, clickY];
              getInVertex = false;
              getInPoly = false;
              MODE = 0;
              captureScene();
              poly.update(clickX, clickY, 0);
              cnt++;
              poly.num++;
            } else {
              state = 'resize';
            }
          } else {
            if (state == 'finish_draw') {// closing
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
              numPoly += 1;
              $('#poly_count').text(numPoly);
              // console.log(numPoly);
              polyLabeling.clearGlobals();

              state = 'select';
              targetPoly = poly;
              polyLabeling.redraw();
              poly.drawHiddenPoly(poly.num);

              $('#category_select').val(poly.category);
              // $("#submit_btn").attr("disabled", true);

              captureScene();
            } else if (state == 'draw') {
              let dis = calcuDis(poly.p[cnt - 1],
                  [clickX, clickY]);
              if (dis > 2) {
                poly.update(clickX, clickY, cnt);
                cnt++;
                poly.num++;
                window.removeEventListener('dblclick',
                    DoubleClick, false);
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
        if (polyLabeling.resize) {
          mouseDownHandler(e);
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
       * @param {type} e: Description.
       */
      function mouseMove(e) {
        curpos = computePosition(e);
        if (curpos[0] < 0 || curpos[1] < 0) return;
        let mouseX = curpos[0];
        let mouseY = curpos[1];

        if (state == 'draw') {
          poly.p[cnt] = [mouseX, mouseY];
          poly.num = cnt;
          let currentDis = MIN_DIS;
          if (magnify) currentDis = Math.round(MIN_DIS / MagRatio);
          if (calcuDis(curpos, startpos) <= currentDis
              && (cnt != 1)) {
            state = 'finish_draw';
            ctx.beginPath();
            ctx.arc(startpos[0], startpos[1],
                6, 0, 2 * Math.PI, false);
            ctx.closePath();
            ctx.fillStyle = VERTEX_FILL;
            ctx.fill();
            if (magnify) {
              hiddenBiggerCtx.beginPath();
              hiddenBiggerCtx.arc(startpos[0], startpos[1],
                  6 / MagRatio, 0, 2 * Math.PI, false);
              hiddenBiggerCtx.closePath();
              hiddenBiggerCtx.fillStyle = VERTEX_FILL;
              hiddenBiggerCtx.fill();
            }
          } else {
            // keep drawing
            ctx.putImageData(lastScene, 0, 0);

            poly.drawPoly(cnt + 1, ctx);
            ctx.beginPath();
            ctx.arc(startpos[0], startpos[1],
                6, 0, 2 * Math.PI, false);
            ctx.closePath();
            ctx.fillStyle = poly.styleColor(0.7);
            ctx.fill();

            if (magnify) {
              hiddenBiggerCtx.putImageData(HidlastScene, 0, 0);
              poly.drawPoly(cnt + 1, hiddenBiggerCtx);
              hiddenBiggerCtx.beginPath();
              hiddenBiggerCtx.arc(startpos[0], startpos[1],
                  6 / MagRatio, 0, 2 * Math.PI, false);
              hiddenBiggerCtx.closePath();
              hiddenBiggerCtx.fillStyle = poly.colors(poly.id,
                  0.7);
              hiddenBiggerCtx.fill();
            }
          }
        } else if (state == 'finish_draw') {
          if (calcuDis(curpos, startpos) > MIN_DIS) {
            poly.p[cnt] = [mouseX, mouseY];
            poly.num = cnt;
            state = 'draw';

            ctx.putImageData(lastScene, 0, 0);
            poly.drawPoly(cnt + 1, ctx);
            if (magnify) {
              hiddenBiggerCtx.putImageData(HidlastScene, 0, 0);
              poly.drawPoly(cnt + 1, hiddenBiggerCtx);
            }
          }
        } else if (state == 'resize' || state == 'select_resize') {
          getInVertex = false;
          getInPoly = false;

          if (MODE == VERTEX) {
            minPoly.p[minPoint] = [mouseX, mouseY];
            polyLabeling.redraw(polylist[minPoly.id]);

            minPoly.changeHiddenVertex(minPoint);
            minPoly.changeHiddenVertex(
                (minPoint - 1 + minPoly.num) % minPoly.num);

            if (state == 'resize') {
              minPoly.drawPoly(minPoly.num, ctx);
              if (magnify) {
                minPoly.drawPoly(minPoly.num, hiddenBiggerCtx);
              }
            } else if (state == 'select_resize') {
              minPoly.drawPoly(minPoly.num, ctx, true);
              if (magnify) {
                minPoly.drawPoly(minPoly.num, hiddenBiggerCtx
                    , true);
              }
            }
          } else if (MODE == MID_VERTEX) {
            if (!IfGenarate) {
              minPoly.genarateNewVertex(minMidPoint);
              IfGenarate = true;
            }

            minPoly.p[minMidPoint + 1] = [mouseX, mouseY];

            minPoly.changeHiddenVertex(minMidPoint);
            minPoly.changeHiddenVertex(
                (minMidPoint + 1) % minPoly.num);
            polyLabeling.redraw(minPoly);

            if (state == 'resize') {
              minPoly.drawPoly(minPoly.num, ctx);
              if (magnify) {
                minPoly.drawPoly(minPoly.num, hiddenBiggerCtx);
              }
            } else if (state == 'select_resize') {
              minPoly.drawPoly(minPoly.num, ctx, true);
              if (magnify) {
                minPoly.drawPoly(minPoly.num, hiddenBiggerCtx
                    , true);
              }
            }
          }
        } else {
          MODE = 0;
          minPoly = polyLabeling.selectPoly(mouseX, mouseY);
          if (state == 'free') {
            minPoint = polyLabeling.selectPolyVertex(mouseX,
                mouseY);
            if (minPoint % 2 == 0) {
              minPoint = minPoint / 2;
              minMidPoint = -1;
            } else {
              minMidPoint = (minPoint - 1) / 2;
              minPoint = -1;
            }
          } else if (state == 'select') {
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
            if (minDis >= 6) {
              minMidPoint = -1;
              minPoint = -1;
              minPoly = polyLabeling.selectPoly(mouseX, mouseY);
            }
          }

          if (!ShowMid) minMidPoint = -1;

          if (minPoly != -1) {
            if (getInPoly == false) {
              minPoly.showControlPoints(minPoly.num, ctx);
              if (magnify) {
                minPoly.showControlPoints(minPoly.num,
                    hiddenBiggerCtx);
              }
              lastPoly = minPoly;
              getInPoly = true;
            }
            if (minPoly != lastPoly) {
              polyLabeling.redraw();
              minPoly.showControlPoints(minPoly.num, ctx);
              if (magnify) {
                minPoly.showControlPoints(minPoly.num,
                    hiddenBiggerCtx);
              }
              lastPoly = minPoly;
            }
            let vdis = MAX_DIS;
            let mdis = MAX_DIS;

            // double check
            if (minPoint != -1 && minPoly.num > minPoint) {
              vdis = calcuDis(curpos, minPoly.p[minPoint]);
            }
            if (minMidPoint != -1 && minPoly.num > minMidPoint) {
              mdis = calcuDis(curpos,
                  minPoly.hidden_p[minMidPoint]);
            }

            if (minPoint != -1 && minMidPoint == -1
                && vdis < 6) {
              MODE = VERTEX;
            } else if (minPoint == -1 && minMidPoint != -1 &&
                mdis < 6) {
              MODE = MID_VERTEX;
            } else if (minPoint != -1 && minMidPoint != -1) {
              if (vdis < mdis && vdis < 6) MODE = VERTEX;
              else if (mdis <= vdis && mdis < 6) {
                MODE = MID_VERTEX;
              }
            } else {
              MODE = 0;
            }

            if (MODE == MID_VERTEX) {
              if (minPoly.num <= minMidPoint
                  || minPoly.num <= 0) {
                MODE = 0;
              } else if (minPoly.degree[minMidPoint] == 2 ||
                  minPoly.degree[(minMidPoint + 1) % minPoly.num]
                  == 2) {
                MODE = 0;
              }
            }

            if (MODE == VERTEX) {
              if (getInVertex == false) {
                if (minPoly.num > minPoint
                    && vdis < 6) {
                  polyLabeling.resize = true;
                  getInVertex = true;
                  polyLabeling.drawSelectVertex(MODE,
                      minPoly.p[minPoint]);
                }
              }
            } else if (MODE == MID_VERTEX) {
              if (getInVertex == false) {
                if (minPoly.num > minMidPoint && mdis < 6) {
                  polyLabeling.resize = true;
                  getInVertex = true;
                  polyLabeling.drawSelectVertex(MODE,
                      minPoly.hidden_p[minMidPoint]);
                }
              }
            } else {
              if (getInVertex == true) {
                getInVertex = false;
                polyLabeling.resize = false;
                ctx.putImageData(lastImg, 0, 0);
                if (magnify) {
                  hiddenBiggerCtx.putImageData(lastHidImg,
                      0, 0);
                }
              }
            }
          } else {// undo
            if (getInVertex == true) {
              // //console.log(getIn);
              getInVertex = false;
              MODE = 0;
              polyLabeling.resize = false;

              ctx.putImageData(lastImg, 0, 0);
              if (magnify) {
                hiddenBiggerCtx.putImageData(lastHidImg, 0, 0);
              }
            }
            if (getInPoly == true) {
              // //console.log(getIn);
              getInPoly = false;
              MODE = 0;
              polyLabeling.resize = false;
              lastPoly = -1;
              polyLabeling.redraw();
            }
          }
        }
        if (magnify) {
          magnifierCtx.globalCompositeOperation = 'destination-over';
          let bbox = mainCanvas.getBoundingClientRect();

          magnifier.style.left
              = (mouseX + bbox.left - magnifier.width / 2) + 'px';
          magnifier.style.top
              = (mouseY + bbox.top - magnifier.height / 2) + 'px';

          let x = mouseX - magnifier.width / (MagRatio * 2);
          let y = mouseY - magnifier.height / (MagRatio * 2);

          if (e.clientX >= bbox.left &&
              e.clientY >= bbox.top
          ) {
            magnifier.style.display = 'inline';

            let drawWidth = magnifier.width / MagRatio;
            let drawHeight = magnifier.height / MagRatio;

            if (y + drawHeight > styleHeight) {
              drawHeight = styleHeight - y;
            }
            if (x + drawWidth > styleWidth) {
              drawWidth = styleWidth - x;
            }
            let startx;
            let starty;
            startx = x;
            starty = y;
            if (x < 0) startx = 0;
            if (y < 0) starty = 0;
            let enlarge = hiddenBiggerCtx.getImageData(
                startx * MagRatio, starty * MagRatio,
                MagRatio * (x + drawWidth),
                MagRatio * (y + drawHeight));

            magnifierCtx.clearRect(0, 0, magnifier.width,
                magnifier.height);
            magnifierCtx.putImageData(enlarge,
                MagRatio * (startx - x), MagRatio * (starty - y));
            magnifierCtx.drawImage(sourceImage, x / ratio,
                y / ratio, drawWidth / ratio, drawHeight / ratio,
                0, 0, MagRatio * drawWidth, MagRatio * drawHeight);
          } else {
            magnifier.style.display = 'none';
            magnifier.style.left = 0 + 'px';
            magnifier.style.top = 0 + 'px';
          }
        }
      }

      /**
       * Summary: To be completed.
       * @param {type} ignoredEvent: Description.
       */
      function mouseUp(ignoredEvent) {
        if (state == 'resize' ||
            state == 'select_resize') {
          MODE = 0;
          IfGenarate = false;
          cnt = 0;
          if (minPoly && minPoly != -1) {
            polylist[minPoly.id] = minPoly;
          }

          if (state == 'resize') {
            polyLabeling.clearGlobals();
            polyLabeling.redraw();
            polyLabeling.hidden_redraw();
          } else {
            state = 'select';
            polyLabeling.redraw();
            polyLabeling.hidden_redraw();
          }
          captureScene();
        }
      }

      /**
       * Summary: To be completed.
       * @param {type} e: Description.
       */
      function KeyDown(e) {
        let keyID = e.KeyCode ? e.KeyCode : e.which;
        if (keyID === 27) {
          if (state == 'draw') {
            cnt = 0;
            state = 'free';
            polyLabeling.redraw();
            captureScene();
            window.addEventListener('dblclick', DoubleClick, false);
          } else if (getInVertex == true &&
              polyLabeling.resize == true && MODE == VERTEX) {
            if (minPoly.degree[minPoint] == 0) {
              minPoly.deleteVertex(minPoint);
            } else {
              minPoly.deleteBezier(minPoint);
            }

            MODE = 0;
            cnt = 0;
            if (minPoly.num > 1) {
              polylist[minPoly.id] = minPoly;
            }

            if (minPoly.num <= 1) {
              if (targetPoly == minPoly) {
                targetPoly = -1;
                state = 'free';
              }
            }
            if (state != 'select') {
              polyLabeling.clearGlobals();
            } else {
              getInVertex = false;
              getInPoly = false;
              polyLabeling.resize = false;
            }

            polyLabeling.redraw();
            polyLabeling.hidden_redraw();
            captureScene();
          } else {
            alert('Nothing to delete!');
          }
        } else if (keyID === 66) {// B
          if (getInVertex == true && polyLabeling.resize == true
              && MODE == MID_VERTEX) {
            minPoly.addBezier(minMidPoint);
            MODE = 0;
            cnt = 0;

            if (state !== 'select') {
              polyLabeling.clearGlobals();
            } else {
              getInVertex = false;
              getInPoly = false;
              polyLabeling.resize = false;
            }

            polylist[minPoly.id] = minPoly;
            polyLabeling.redraw();
            polyLabeling.hidden_redraw();

            captureScene();
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
        } else if (keyID === 77) {// M
          if (magnify == false) {
            polyLabeling.startMagnifier();
          } else {
            polyLabeling.removeMagnifier();
          }
        } else if (keyID === 46 || keyID === 8) {
          if (state !== 'draw' && getInVertex == true &&
              polyLabeling.resize == true && MODE == VERTEX) {
            if (minPoly.degree[minPoint] == 0) {
              minPoly.deleteVertex(minPoint);
            } else {
              minPoly.deleteBezier(minPoint);
            }
            MODE = 0;
            cnt = 0;
            if (minPoly.num > 1) {
              polylist[minPoly.id] = minPoly;
            }

            if (minPoly.num <= 1) {
              if (minPoly == targetPoly) {
                targetPoly = -1;
                state = 'free';
              }
            }
            if (state != 'select') {
              polyLabeling.clearGlobals();
            } else {
              getInVertex = false;
              getInPoly = false;
              polyLabeling.resize = false;
            }

            polyLabeling.redraw();
            polyLabeling.hidden_redraw();

            captureScene();
          } else if (state == 'select') {
            if (targetPoly != -1) {
              let selectId = targetPoly.id;
              polylist.splice(selectId, 1);
              numPoly -= 1;
              $('#poly_count').text(numPoly);
              for (let i = selectId; i < polylist.length; i++) {
                polylist[i].id--;
              }
              polyLabeling.clearGlobals();
              polyLabeling.redraw();
              polyLabeling.hidden_redraw();
              cnt = 0;
              captureScene();
            } else {
              alert('Please select the object you want to'
                  + ' delete.');
            }
            // $("#clear_btn").attr("disabled", false);
            // $("#submit_btn").attr("disabled", false);
          } else {
            alert('Nothing to delete!');
          }
        } else if (keyID === 72) {
          if (state == 'free' || state == 'select') {
            if (labelShowtime === 0) {
              labelShowtime = 1;
              polyLabeling.redraw();
              $('#label_btn').text('Hide Label (L)');
              if (state === 'select') {
                getInVertex = false;
                getInPoly = false;
                polyLabeling.resize = false;
              }
              captureScene();
            } else {
              labelShowtime = 0;
              polyLabeling.redraw();
              $('#label_btn').text('Show Label (L)');
              if (state === 'select') {
                getInVertex = false;
                getInPoly = false;
                polyLabeling.resize = false;
              }
              captureScene();
            }
          }
        }
      }

      /**
       * Summary: To be completed.
       * @param {type} e: Description.
       */
      function DoubleClick(e) {
        let pos = computePosition(e);
        let mouseX = pos[0];
        let mouseY = pos[1];
        if (pos[0] < 0 || pos[1] < 0) {
          if (state === 'select') {
            polyLabeling.clearGlobals();
            polyLabeling.redraw();
            captureScene();
          }
          return;
        }
        // avoid user hit more than twice
        window.removeEventListener('mousedown', mouseDown, false);
        selectedPoly = polyLabeling.selectPoly(mouseX, mouseY);
        clearTimeout(interval);
        clearTimeout(enable);

        if (state == 'draw') {
          if (cnt > 1 && poly
              && calcuDis(poly.p[cnt - 2], pos) > MIN_DIS) {
            window.addEventListener('mousedown', mouseDown, false);
            return;
          }
          cnt = 0;
          state = 'free';
          polyLabeling.redraw();
          captureScene();
        }
        if (selectedPoly !== -1) {
          state = 'select';
          // $("#submit_btn").attr("disabled", true);
          cnt = 0;
          if (selectedPoly !== targetPoly) {
            targetPoly = selectedPoly;
            polyLabeling.redraw();
            $('#category_select').val(selectedPoly.category);
          }
        } else {
          polyLabeling.clearGlobals();
          polyLabeling.redraw();
          captureScene();
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

      $('#pagination_control').mouseover(function() {
        window.removeEventListener('mousedown', mouseDown, false);
        window.removeEventListener('mousemove', mouseMove, false);
        window.removeEventListener('mouseup', mouseUp, false);
        window.removeEventListener('dblclick', DoubleClick, false);
      });
      $('#pagination_control').mouseleave(function() {
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

      $('#delete_btn').click(function() {
        // ctx.putImageData(lastScene,0,0);
        if (targetPoly != -1) {
          let selectId = targetPoly.id;
          polylist.splice(selectId, 1);
          numPoly -= 1;
          $('#poly_count').text(numPoly);
          for (let i = selectId; i < polylist.length; i++) {
            polylist[i].id--;
          }

          polyLabeling.clearGlobals();
          polyLabeling.redraw();
          polyLabeling.hidden_redraw();
          cnt = 0;
          captureScene();
        } else {
          alert('Please select the object you want to delete.');
        }
        // /$("#clear_btn").attr("disabled", false);
        // $("#submit_btn").attr("disabled", false);
      });
      $('#shape_btn').click(function() {
        if (magnify == false) {
          polyLabeling.startMagnifier();
        } else {
          polyLabeling.removeMagnifier();
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
              polyLabeling.resize = false;
            }
            captureScene();
          } else {
            labelShowtime = 0;
            polyLabeling.redraw();
            $('#label_btn').text('Show Label (L)');
            if (state === 'select') {
              getInVertex = false;
              getInPoly = false;
              polyLabeling.resize = false;
            }
            captureScene();
          }
        }
      });

      $('#category_select').change(function() {
        if (state == 'select') {
          let catIdx = document.getElementById('category_select').selectedIndex;
          let cate = assignment.category[catIdx];
          cnt = 0;
          if (targetPoly !== -1) {
            polylist[targetPoly.id].category = cate;
            let pos = calcuCenter(targetPoly);
            polyLabeling.redraw();
            if (labelShowtime === 0) {
              targetPoly.fillLabel(pos);
            }
          }
          getInVertex = false;
          getInPoly = false;
          polyLabeling.resize = false;
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
     * @param {type} fixedLabel: Description.
     * @param {type} id: Description.
     */
    function Poly(fixedLabel, id) {
      SatLabel.call(this, null, id);
      this.num = 0;
      this.beziernum = 0;
      this.p = [];
      this.degree = [];
      this.BezierOffset = [];
      this.hidden_p = [];
      this.category = fixedLabel;
    }

    Poly.prototype = Object.create(SatLabel.prototype);

    Poly.prototype.update = function(clickX, clickY, cnt) {
      let vec = [clickX, clickY];
      let prevX;
      let prevY;
      this.p[cnt] = vec;
      this.degree[cnt] = 0;

      if (cnt > 0) {
        prevX = this.p[cnt - 1][0];
        prevY = this.p[cnt - 1][1];
        this.hidden_p[cnt - 1] = [
          (clickX + prevX) / 2,
          (clickY + prevY) / 2];
      }
      return true;
    };

    Poly.prototype.computeDegree = function() {
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
    };
    Poly.prototype.deleteBezier = function(index) {
      let offset = -1;
      for (let i = 0; i < this.beziernum; i++) {
        if (this.BezierOffset[i] <= index &&
            this.BezierOffset[i] + 3 >= index) {
          offset = i;
          break;
        } else if (index == 0 && this.BezierOffset[i] + 3
            == this.num) {
          offset = i;
          break;
        }
      }
      if (offset == -1) {
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
    };

    Poly.prototype.drawPoly = function(cnt, context, select) {
      let width = 2;
      let radius = 3;
      if (context === hiddenBiggerCtx) {
        width = width / MagRatio;
        radius = radius / MagRatio;
      }

      let withBezier = false;
      if (this.beziernum > 0) withBezier = true;
      if (withBezier == false) {
        this.drawPolyWithoutBezier(cnt, context);
      } else if (withBezier == true && cnt > 4) {
        this.drawPolyWithBezier(cnt, context);
      }
      context.closePath();

      if (select) {
        context.fillStyle = SELECT_COLOR;
      } else {
        context.fillStyle = this.styleColor(ALPHA);
      }
      context.fill();
      context.lineWidth = width;
      context.strokeStyle = this.styleColor(1);
      context.stroke();

      if (withBezier && cnt == 4) {
        let index = this.BezierOffset[0];
        context.beginPath();
        context.moveTo(this.p[index][0], this.p[index][1]);
        context.bezierCurveTo(this.p[(index + 1) % cnt][0],
            this.p[(index + 1) % cnt][1],
            this.p[(index + 2) % cnt][0], this.p[(index + 2) % cnt][1],
            this.p[(index + 3) % cnt][0], this.p[(index + 3) % cnt][1]);
      }
      context.strokeStyle = this.styleColor(1);
      context.stroke();

      if (withBezier) {
        this.drawDashLine(cnt, context);
      }

      context.fillStyle = this.styleColor(1);
      for (let j = 0; j < cnt; j++) {
        context.beginPath();
        context.arc(this.p[j][0], this.p[j][1], radius, 0,
            2 * Math.PI, false);
        context.closePath();
        context.fill();
      }

      if (labelShowtime) {
        let tmpPoly = this;
        let pos = calcuCenter(tmpPoly);
        this.fillLabel(pos);
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
      if (num == 0) return;
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
        if (nextIndex != (index + 4) % cnt &&
            nextIndex != (index + 3) % cnt) {
          for (let j = (index + 4) % cnt;
               j != nextIndex; j = (j + 1) % cnt) {
            context.lineTo(this.p[j][0], this.p[j][1]);
          }
          context.lineTo(this.p[nextIndex][0], this.p[nextIndex][1]);
        } else if (nextIndex == (index + 4) % cnt) {
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

      if (index != (nextIndex + 4) % cnt
          && index != (nextIndex + 3) % cnt) {
        for (let j = (nextIndex + 4) % cnt; j != index;
             j = (j + 1) % cnt) {
          context.lineTo(this.p[j][0], this.p[j][1]);
        }
        context.lineTo(this.p[index][0], this.p[index][1]);
      } else if (index == (nextIndex + 4) % cnt) {
        context.lineTo(this.p[index][0], this.p[index][1]);
      }
    };

    Poly.prototype.drawDashLine = function(cnt, context) {
      let num = this.beziernum;
      if (num == 0) return;
      let index = this.BezierOffset[0];
      context.setLineDash([3]);

      for (let i = 0; i < num; i++) {
        index = this.BezierOffset[i];
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

    Poly.prototype.fillLabel = function(pos) {
      let word = this.category.substring(0, 3);
      let tagWidth = word.length;
      let tagHeight = 15;
      ctx.fillStyle = this.styleColor(1);
      ctx.fillRect(pos[0] - tagWidth, pos[1]
          - 2, tagWidth * 9, tagHeight);
      ctx.fillStyle = 'rgb(0, 0, 0)';
      ctx.font = '13px Verdana';
      ctx.fillText(word, pos[0] - tagWidth, pos[1] + tagHeight
          - 3);
    };

    Poly.prototype.drawHiddenPoly = function(cnt) {
      hiddenCtx.fillStyle = this.hidden_colors(this.id, -1, -1);
      hiddenCtx.strokeStyle = this.hidden_colors(this.id, -1, -1);

      let withBezier = false;
      if (this.beziernum > 0) withBezier = true;
      if (withBezier == false) {
        this.drawPolyWithoutBezier(cnt, hiddenCtx);
      } else if (withBezier == true && cnt > 4) {
        this.drawPolyWithBezier(cnt, hiddenCtx);
      }
      hiddenCtx.closePath();
      hiddenCtx.fill();
      hiddenCtx.lineWidth = 2;
      hiddenCtx.stroke();

      if (withBezier && cnt == 4) {
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
        polylist.splice(this.id, 1);
        numPoly -= 1;
        $('#poly_count').text(numPoly);
        this.hidden_p = [];
        for (let i = this.id; i < polylist.length; i++) {
          polylist[i].id--;
        }
        return;
      }
      for (let i = 0; i < this.beziernum; i++) {
        if (this.BezierOffset[i] > index) {
          this.BezierOffset[i]--;
        }
      }
      let prev = (index - 1 + this.num) % this.num;

      this.changeHiddenVertex(prev);
    };

    Poly.prototype.changeHiddenVertex = function(index) {
      let next = (index + 1) % this.num;
      index = index % this.num;

      this.hidden_p[index] = [
        (this.p[index][0] +
            this.p[next][0]) / 2, (this.p[index][1] +
            this.p[next][1]) / 2];
    };

    Poly.prototype.drawHiddenVertex = function(cnt) {
      if (this.num > 1 && ShowMid) {
        for (let i = 0; i < cnt; i++) {
          if (cnt == 2 && i == 1) break;
          if (this.degree[i] < 2 && this.degree[(i + 1) % cnt] < 2) {
            hiddenCtx.beginPath();
            if (calcuDis(this.hidden_p[i], this.p[i]) > MIN_DIS * 2
                && calcuDis(this.hidden_p[i], this.p[(i + 1) % cnt])
                > MIN_DIS * 2) {
              hiddenCtx.arc(this.hidden_p[i][0],
                  this.hidden_p[i][1], 6, 0, 2 * Math.PI, false);
            } else {
              hiddenCtx.arc(this.hidden_p[i][0],
                  this.hidden_p[i][1], 3, 0, 2 * Math.PI, false);
            }
            hiddenCtx.closePath();
            hiddenCtx.fillStyle
                = this.hidden_colors(-1, 2 * i + 1, this.id);
            hiddenCtx.fill();
          }
        }
      }

      for (let j = 0; j < cnt; j++) {
        hiddenCtx.beginPath();
        hiddenCtx.arc(this.p[j][0], this.p[j][1],
            6, 0, 2 * Math.PI, false);
        hiddenCtx.closePath();
        hiddenCtx.fillStyle = this.hidden_colors(-1, 2 * j, this.id);
        hiddenCtx.fill();
      }
    };

    Poly.prototype.showControlPoints = function(cnt, context) {
      let radius = 3;
      if (context === hiddenBiggerCtx) radius = radius / MagRatio;

      context.fillStyle = this.styleColor(0.5);
      for (let j = 0; j < cnt; j++) {
        context.beginPath();
        context.arc(this.p[j][0], this.p[j][1],
            radius + 2, 0, 2 * Math.PI, false);
        context.closePath();
        context.fill();
      }

      context.fillStyle = MID_CONTROL_POINT;
      for (let j = 0; j < cnt; j++) {
        if (this.degree[j] < 2 && this.degree[(j + 1) % cnt] < 2
            && ShowMid) {
          context.beginPath();
          context.arc(this.hidden_p[j][0], this.hidden_p[j][1],
              radius, 0, 2 * Math.PI, false);
          context.closePath();
          context.fill();
        }
      }

      if (this.beziernum > 0) {
        context.fillStyle = BEZIER_COLOR;
        for (let i = 0; i < this.beziernum; i++) {
          for (let j = this.BezierOffset[i];
               j != (this.BezierOffset[i] + 4) % this.num;
               j = (j + 1) % this.num) {
            context.beginPath();
            context.arc(this.p[j][0], this.p[j][1],
                radius, 0, 2 * Math.PI, false);
            context.closePath();
            context.fill();
          }
        }
      }
    };
    Poly.prototype.hidden_colors = function(x, y, z) {
      return 'rgb(' + (x + 1) + ',' + (y + 1) + ',' + (z + 1) + ')';
    };
    return Poly;
  })();
}).call(this);
