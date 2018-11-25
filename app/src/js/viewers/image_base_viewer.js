import $ from 'jquery';
import {BaseViewer} from './base_viewer';
/* :: import {BaseController} from '../controllers/base_controller'; */
import type {ImageViewerConfigType} from '../functional/types';
import {sprintf} from 'sprintf-js';
import Session from '../common/session_single';

/**
 * BaseViewer2D class
 */
export class BaseViewer2D extends BaseViewer {
  divCanvas: Object;
  canvas: Object;
  context: Object;
  MAX_SCALE: number;
  MIN_SCALE: number;
  SCALE_RATIO: number;
  UP_RES_RATIO: number;
  scale: number;
  // need these two below to prevent jitters caused by round off
  canvasHeight: number;
  canvasWidth: number;
  displayToImageRatio: number;
  // False for image canvas, true for anything else
  upRes: boolean;
  // TODO: This can be more general for different types of view composition
  isAssistantView: boolean;

  /**
   * @param {BaseController} controller
   * @param {string} canvasId
   * @param {string} canvasSuffix
   * @param {boolean} upRes
   * @constructor
   */
  constructor(controller/* : BaseController */, canvasId: string,
              canvasSuffix: string = '', upRes: boolean = true) {
    super(controller);
    // necessary variables
    let divCanvasName = 'div-canvas';
    if (canvasSuffix !== '') {
      divCanvasName = sprintf('%s-%s', divCanvasName, canvasSuffix);
      canvasId = sprintf('%s-%s', canvasId, canvasSuffix);
    }
    let divCanvas = document.getElementById(divCanvasName);
    if (divCanvas) {
      this.divCanvas = divCanvas;
    }
    let canvas = document.getElementById(canvasId);
    if (canvas) {
      this.canvas = canvas;
    }
    // $FlowFixMe
    this.context = this.canvas.getContext('2d');
    this.MAX_SCALE = 3.0;
    this.MIN_SCALE = 1.0;
    this.SCALE_RATIO = 1.05;
    this.UP_RES_RATIO = 2;
    this.scale = 1;

    this.isAssistantView = false;
    this.upRes = upRes;
  }

  /**
   * Convert image coordinate to canvas coordinate.
   * If affine, assumes values to be [x, y]. Otherwise
   * performs linear transformation.
   * @param {Array<number>} values - the values to convert.
   * @return {Array<number>} - the converted values.
   */
  toCanvasCoords(values: Array<number>) {
    if (values) {
      for (let i = 0; i < values.length; i++) {
        values[i] *= this.displayToImageRatio;
        if (this.upRes) {
          values[i] *= this.UP_RES_RATIO;
        }
      }
    }
    return values;
  }

  /**
   * Convert canvas coordinate to image coordinate.
   * If affine, assumes values to be [x, y]. Otherwise
   * performs linear transformation.
   * @param {Array<number>} values - the values to convert.
   * @return {Array<number>} - the converted values.
   */
  toImageCoords(values: Array<number>) {
    if (values) {
      for (let i = 0; i < values.length; i++) {
        values[i] /= this.displayToImageRatio;
      }
    }
    return values;
  }

  /**
   * get visible canvas coords
   * @return {Array<number>}
   */
  getVisibleCanvasCoords() {
    let imgRect = this.canvas.getBoundingClientRect();
    let divRect = this.divCanvas.getBoundingClientRect();
    return [divRect.x - imgRect.x, divRect.y - imgRect.y];
  }

  /**
   * Get the mouse position on the canvas in the image coordinates.
   * @param {Object} e: mouse event
   * @return {CoordinateType}: mouse position (x,y) on the canvas
   */
  getMousePos(e: Object) {
    // limit mouse within the image
    let rect = this.canvas.getBoundingClientRect();
    let x = Math.min(
      Math.max(e.clientX, rect.x),
      rect.x + this.canvasWidth);
    let y = Math.min(
      Math.max(e.clientY, rect.y),
      rect.y + this.canvasHeight);
    // limit mouse within the main div
    let rectDiv = this.divCanvas.getBoundingClientRect();
    x = Math.min(
      Math.max(x, rectDiv.x),
      rectDiv.x + rectDiv.width
    );
    y = Math.min(
      Math.max(y, rectDiv.y),
      rectDiv.y + rectDiv.height
    );
    return {
      x: (x - rect.x) / this.displayToImageRatio,
      y: (y - rect.y) / this.displayToImageRatio,
    };
  }

  /**
   * Set the scale of the image in the display
   * @param {Array<number>} mouseOffset: [x, y]
   */
  updateScale(mouseOffset: Array<number> = []) {
    let config: ImageViewerConfigType = this.getCurrentViewerConfig();
    let upperLeftCoords = [0, 0];
    let rectDiv = this.divCanvas.getBoundingClientRect();
    if (config.viewScale > 1.0) {
      upperLeftCoords = this.getVisibleCanvasCoords();
      if (mouseOffset.length !== 2) {
        mouseOffset = [
          Math.min(rectDiv.width, this.canvas.width) / 2,
          Math.min(rectDiv.height, this.canvas.height) / 2,
        ];
      } else {
        mouseOffset = this.toCanvasCoords(mouseOffset);
        mouseOffset[0] -= upperLeftCoords[0];
        mouseOffset[1] -= upperLeftCoords[1];
      }
    }

    // set scale
    if (config.viewScale >= this.MIN_SCALE
      && config.viewScale < this.MAX_SCALE) {
      let ratio = config.viewScale / this.scale;
      this.context.scale(ratio, ratio);
    } else {
      return;
    }
    // handle buttons
    if (config.viewScale >= this.MIN_SCALE * this.SCALE_RATIO) {
      $('#decrease-btn').prop('disabled', false);
    } else {
      $('#decrease-btn').prop('disabled', true);
    }
    if (config.viewScale <= this.MAX_SCALE / this.SCALE_RATIO) {
      $('#increase-btn').prop('disabled', false);
    } else {
      $('#increase-btn').prop('disabled', true);
    }
    // resize canvas
    let item = this.getCurrentItem();
    let image = Session.images[item.index];
    let ratio = image.width / image.height;

    if (rectDiv.width / rectDiv.height > ratio) {
      this.canvasHeight = rectDiv.height * config.viewScale;
      this.canvasWidth = this.canvasHeight * ratio;
      this.displayToImageRatio = this.canvasHeight / image.height;
    } else {
      this.canvasWidth = rectDiv.width * config.viewScale;
      this.canvasHeight = this.canvasWidth / ratio;
      this.displayToImageRatio = this.canvasWidth / image.width;
    }

    // translate back to origin
    if (mouseOffset) {
      this.divCanvas.scrollTop = this.canvas.offsetTop;
      this.divCanvas.scrollLeft = this.canvas.offsetLeft;
    }

    // set canvas resolution
    if (this.upRes) {
      this.canvas.height = this.canvasHeight * this.UP_RES_RATIO;
      this.canvas.width = this.canvasWidth * this.UP_RES_RATIO;
    } else {
      this.canvas.height = this.canvasHeight;
      this.canvas.width = this.canvasWidth;
    }

    // set canvas size
    this.canvas.style.height = this.canvasHeight + 'px';
    this.canvas.style.width = this.canvasWidth + 'px';

    // set padding
    let padding = this._getPadding();
    let padX = padding.x;
    let padY = padding.y;

    this.canvas.style.left = padX + 'px';
    this.canvas.style.top = padY + 'px';

    // zoom to point
    if (mouseOffset) {
      if (this.canvasWidth > rectDiv.width) {
        this.divCanvas.scrollLeft =
          config.viewScale / this.scale * (upperLeftCoords[0] + mouseOffset[0])
          - mouseOffset[0];
      }
      if (this.canvasHeight > rectDiv.height) {
        this.divCanvas.scrollTop =
          config.viewScale / this.scale * (upperLeftCoords[1] + mouseOffset[1])
          - mouseOffset[1];
      }
    }
    this.scale = config.viewScale;
  }

  /**
   * Get the padding for the image given its size and canvas size.
   * @return {object} padding
   */
  _getPadding() {
    let rectDiv = this.divCanvas.getBoundingClientRect();
    return {
      x: Math.max(0, (rectDiv.width - this.canvasWidth) / 2),
      y: Math.max(0, (rectDiv.height - this.canvasHeight) / 2),
    };
  }

  /**
   * Redraw the image canvas
   * @return {boolean}: whether redraw is successful
   */
  redraw(): boolean {
    if (!super.redraw()) {
      return false;
    }
    if (!this.getCurrentItem().loaded) {
      return false;
    }
    this.resizeCanvas();
    this.updateScale();
    return true;
  }

  /**
   * Resize the image canvas
   */
  resizeCanvas(): void {
    let config: ImageViewerConfigType = this.getCurrentViewerConfig();
    // TODO: make this configurable
    let sideBarWidth = this.state.layout.toolbarWidth;
    let windowSize = $(window).width() - 11 - sideBarWidth;
    let splitBarWidth = 4;
    let newWidth = 0;
    if (this.state.layout.assistantView) {
      windowSize -= splitBarWidth;
      let ratio = this.isAssistantView ? this.state.layout.assistantViewRatio :
          1 - this.state.layout.assistantViewRatio;
      newWidth = Math.round(windowSize * ratio);
    } else {
      newWidth = this.isAssistantView ? 0 : windowSize;
    }
    this.divCanvas.style.width = sprintf('%dpx', newWidth);
    this.setCanvasSize(config);
  }

  /**
   * Set the image canvas size
   * @param {ImageViewerConfigType} config: imageViewer configuration
   */
  setCanvasSize(config: ImageViewerConfigType): void {
    let rectDiv = this.divCanvas.getBoundingClientRect();
    this.canvas.style.height = sprintf('%dpx',
        Math.round(rectDiv.height * config.viewScale));
    this.canvas.style.width = sprintf('%dpx',
        Math.round(rectDiv.width * config.viewScale));

    this.canvas.width = rectDiv.width * config.viewScale;
    this.canvas.height = rectDiv.height * config.viewScale;
  }
}
