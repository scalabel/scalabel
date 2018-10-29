import $ from 'jquery';
import {BaseViewer} from './base_viewer';
import {BaseController} from '../controllers/base_controller';
import type {ImageViewerConfigType} from '../functional/types';

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
  displayToImageRatio: number;

  /**
   * @param {BaseController} controller
   * @param {string} canvasId
   * @constructor
   */
  constructor(controller: BaseController, canvasId: string) {
    super(controller);
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
    this.context = this.canvas.getContext('2d');
    this.MAX_SCALE = 3.0;
    this.MIN_SCALE = 1.0;
    this.SCALE_RATIO = 1.05;
    this.UP_RES_RATIO = 2;
    this.scale = 1;
  }

  // /**
  //  * set initial canvas scales
  //  */
  // init() {
  //   // this.store.subscribe(this.redraw.bind(this));
  // }
  //
  // /**
  //  * load the given Item
  //  * @param {number} index
  //  */
  // loaded(index: number) {
  //   let activeItem = this.getActiveItem();
  //   if (activeItem === index) {
  //     this.redraw();
  //   }
  // }
  //
  // /**
  //  * Get current active item
  //  * @return {*}
  //  */
  // getActiveItem(): number {
  //   let state = this.store.getState().present;
  //   return state.current.item;
  // }

  /**
   * Convert image coordinate to canvas coordinate.
   * If affine, assumes values to be [x, y]. Otherwise
   * performs linear transformation.
   * @param {[number]} values - the values to convert.
   * @param {boolean} affine - whether or not this transformation is affine.
   * @return {[number]} - the converted values.
   */
  toCanvasCoords(values: Array<number>, affine: boolean = true): Array<number> {
    let padBox = this._getPadding();
    if (values) {
      for (let i = 0; i < values.length; i++) {
        values[i] *= this.displayToImageRatio * this.UP_RES_RATIO;
      }
    }
    if (affine) {
      values[0] += padBox.x;
      values[1] += padBox.y;
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
    let padBox = this._getPadding();
    if (affine) {
      values[0] -= padBox.x;
      values[1] -= padBox.y;
    }
    if (values) {
      for (let i = 0; i < values.length; i++) {
        values[i] /= this.displayToImageRatio;
      }
    }
    return values;
  }

  /**
   * Update the scale of the image in the display
   */
  updateScale() {
    let config: ImageViewerConfigType = this.getCurrentViewerConfig();
    // set scale
    if (config.viewScale >= this.MIN_SCALE &&
        config.viewScale < this.MAX_SCALE) {
      let ratio = this.scale / config.viewScale;
      this.context.scale(ratio, ratio);
      this.scale = config.viewScale;
    } else {
      return;
    }
    // handle buttons
    // TODO: This should be in TitleBarViewer
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
    let rectDiv = this.divCanvas.getBoundingClientRect();
    this.canvas.style.height =
      Math.round(rectDiv.height * config.viewScale) + 'px';
    this.canvas.style.width =
      Math.round(rectDiv.width * config.viewScale) + 'px';

    this.canvas.width = rectDiv.width * config.viewScale;
    this.canvas.height = rectDiv.height * config.viewScale;
  }

  /**
   * Get the padding for the image given its size and canvas size.
   * @return {Object}: padding box (x,y,w,h)
   */
  _getPadding(): Object {
    let config: ImageViewerConfigType = this.getCurrentViewerConfig();
    // which dim is bigger compared to canvas
    let xRatio = config.imageWidth / this.canvas.width;
    let yRatio = config.imageHeight / this.canvas.height;
    // use ratios to determine how to pad
    let box = {x: 0, y: 0, w: 0, h: 0};
    if (xRatio >= yRatio) {
      this.displayToImageRatio =
          this.canvas.width / config.imageWidth;
      box.x = 0;
      box.y = 0.5 * (this.canvas.height -
        config.imageHeight * this.displayToImageRatio);
      box.w = this.canvas.width;
      box.h = this.canvas.height - 2 * box.y;
    } else {
      this.displayToImageRatio =
          this.canvas.height / config.imageHeight;
      box.x = 0.5 * (this.canvas.width -
        config.imageWidth * this.displayToImageRatio);
      box.y = 0;
      box.w = this.canvas.width - 2 * box.x;
      box.h = this.canvas.height;
    }
    return box;
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
    this.updateScale();
    return true;
  }

  // /**
  //  * incHandler
  //  */
  // _incHandler() {
  //   this.setScale(this.getCurrentViewerConfig().scale *
  // this.getCurrentViewerConfig().SCALE_RATIO);
  //   this.redraw();
  // }
  //
  // /**
  //  * decHandler
  //  */
  // _decHandler() {
  //   this.setScale(this.getCurrentViewerConfig().scale /
  // this.getCurrentViewerConfig().SCALE_RATIO);
  //   this.redraw();
  // }
}
