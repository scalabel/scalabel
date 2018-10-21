import $ from 'jquery';

// TODO: Make this part of Viewer state
const UP_RES_RATIO = 2;

/**
 * ImageBaseViewer class
 */
export class ImageBaseViewer {
  store: Object;
  images: Array<Image>;
  divCanvas: Object;
  canvas: Object;
  ctx: Object;
  MAX_SCALE: number;
  MIN_SCALE: number;
  SCALE_RATIO: number;
  scale: number;
  padBox: Object;
  displayToImageRatio: number;

  /**
   * @param {Object} store
   * @param {Array<Image>} images
   * @param {string} canvasId
   * @constructor
   */
  constructor(store: Object, images: Array<Image>, canvasId: string) {
    this.store = store;
    this.images = images;
    // necessary variables
    let divCanvas = document.getElementById('div_canvas');
    if (divCanvas) {
      this.divCanvas = divCanvas;
    }
    let canvas = document.getElementById(canvasId);
    if (canvas) {
      this.canvas = canvas;
    }
    // $FlowFixMe
    this.ctx = this.canvas.getContext('2d');
    this.MAX_SCALE = 3.0;
    this.MIN_SCALE = 1.0;
    this.SCALE_RATIO = 1.05;
    this.scale = 1.0;
    // Resize
    this.padBox;
    this.displayToImageRatio;
  }

  /**
   * set initial canvas scales
   */
  init() {
    this.setScale(1.0);
    this.store.subscribe(this.redraw.bind(this));
  }

  /**
   * load the given Item
   * @param {number} index
   */
  loaded(index: number) {
    let activeItem = this.getActiveItem();
    if (activeItem === index) {
      this.redraw();
    }
  }

  /**
   * Get current active item
   * @return {*}
   */
  getActiveItem(): number {
    let state = this.store.getState().present;
    return state.current.item;
  }

  /**
   * Convert image coordinate to canvas coordinate.
   * If affine, assumes values to be [x, y]. Otherwise
   * performs linear transformation.
   * @param {[number]} values - the values to convert.
   * @param {boolean} affine - whether or not this transformation is affine.
   * @return {[number]} - the converted values.
   */
  toCanvasCoords(values: Array<number>, affine: boolean = true): Array<number> {
    this.padBox = this._getPadding();
    if (values) {
      for (let i = 0; i < values.length; i++) {
        values[i] *= this.displayToImageRatio * UP_RES_RATIO;
      }
    }
    if (affine) {
      values[0] += this.padBox.x * UP_RES_RATIO;
      values[1] += this.padBox.y * UP_RES_RATIO;
    }
    return values;
  }

  /**
   * Convert canvas coordinate to image coordinate.
   * If affine, assumes values to be [x, y]. Otherwise
   * performs linear transformation.
   * @param {[number]} values - the values to convert.
   * @param {boolean} affine - whether or not this transformation is affine.
   * @return {[number]} - the converted values.
   */
  toImageCoords(values: Array<number>, affine: boolean = true): Array<number> {
    this.padBox = this._getPadding();
    if (affine) {
      values[0] -= this.padBox.x;
      values[1] -= this.padBox.y;
    }
    if (values) {
      for (let i = 0; i < values.length; i++) {
        values[i] /= this.displayToImageRatio;
      }
    }
    return values;
  }

  /**
   * Set the scale of the image in the display
   * @param {number} scale
   */
  setScale(scale: number) {
    // set scale
    if (scale >= this.MIN_SCALE && scale < this.MAX_SCALE) {
      let ratio = scale / this.scale;
      this.ctx.scale(ratio, ratio);
      this.scale = scale;
    } else {
      return;
    }
    // handle buttons
    if (this.scale >= this.MIN_SCALE * this.SCALE_RATIO) {
      $('#decrease-btn').prop('disabled', false);
    } else {
      $('#decrease-btn').prop('disabled', true);
    }
    if (this.scale <= this.MAX_SCALE / this.SCALE_RATIO) {
      $('#increase-btn').prop('disabled', false);
    } else {
      $('#increase-btn').prop('disabled', true);
    }
    // resize canvas
    let rectDiv = this.divCanvas.getBoundingClientRect();
    this.canvas.style.height =
      Math.round(rectDiv.height * this.scale) + 'px';
    this.canvas.style.width =
      Math.round(rectDiv.width * this.scale) + 'px';

    this.canvas.width = rectDiv.width * this.scale;
    this.canvas.height = rectDiv.height * this.scale;
  }

  /**
   * Get the padding for the image given its size and canvas size.
   * @return {Object}: padding box (x,y,w,h)
   */
  _getPadding(): Object {
    let index = this.getActiveItem();
    let image = this.images[index];
    // which dim is bigger compared to canvas
    let xRatio = image.width / this.canvas.width;
    let yRatio = image.height / this.canvas.height;
    // use ratios to determine how to pad
    let box = {x: 0, y: 0, w: 0, h: 0};
    if (xRatio >= yRatio) {
      this.displayToImageRatio = this.canvas.width / image.width;
      box.x = 0;
      box.y = 0.5 * (this.canvas.height -
        image.height * this.displayToImageRatio);
      box.w = this.canvas.width;
      box.h = this.canvas.height - 2 * box.y;
    } else {
      this.displayToImageRatio = this.canvas.height / image.height;
      box.x = 0.5 * (this.canvas.width -
        image.width * this.displayToImageRatio);
      box.y = 0;
      box.w = this.canvas.width - 2 * box.x;
      box.h = this.canvas.height;
    }
    return box;
  }

  /**
   * Redraw the image canvas.
   */
  redraw() {}

  /**
   * incHandler
   */
  _incHandler() {
    this.setScale(this.scale * this.SCALE_RATIO);
    this.redraw();
  }

  /**
   * decHandler
   */
  _decHandler() {
    this.setScale(this.scale / this.SCALE_RATIO);
    this.redraw();
  }
}
