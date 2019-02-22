import React from 'react';
import Session from '../common/session';
import type {ImageViewerConfigType}
from '../functional/types';
import {withStyles} from '@material-ui/core/styles/index';

const pad = 10;

const styles = () => ({
  canvas: {
    position: 'relative',
  },
});

type Props = {
  classes: Object,
  theme: Object,
  height: number,
  width: number,
}

/**
 * Get the current item in the state
 * @return {ItemType}
 */
function getCurrentItem() {
  let state = Session.getState();
  return state.items[state.current.item];
}

/**
 * Retrieve the current viewer configuration
 * @return {ViewerConfigType}
 */
function getCurrentViewerConfig() {
  let state = Session.getState();
  return state.items[state.current.item].viewerConfig;
}

/**
 * Canvas Viewer
 */
class ImageView extends React.Component<Props> {
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
  /**
   * Constructor, handles subscription to store
   * @param {Object} props: react props
   */
  constructor(props: Object) {
    super(props);

    this.MAX_SCALE = 3.0;
    this.MIN_SCALE = 1.0;
    this.SCALE_RATIO = 1.05;
    this.UP_RES_RATIO = 2;
    this.scale = 1;

    this.upRes = true;
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
   * Get the padding for the image given its size and canvas size.
   * @return {object} padding
   */
  _getPadding() {
    return {
      x: Math.max(pad, (this.props.width - this.canvasWidth) / 2),
      y: Math.max(pad, (this.props.height - this.canvasHeight) / 2),
    };
  }

  /**
   * Set the scale of the image in the display
   */
  updateScale() {
    let config: ImageViewerConfigType = getCurrentViewerConfig();

    // set scale
    if (config.viewScale >= this.MIN_SCALE
      && config.viewScale < this.MAX_SCALE) {
      let ratio = config.viewScale / this.scale;
      this.context.scale(ratio, ratio);
    } else {
      return;
    }

    // resize canvas
    let item = getCurrentItem();
    let image = Session.images[item.index];
    let ratio = (image.width + 2 * pad) / (image.height + 2 * pad);

    if (this.props.width / this.props.height > ratio) {
      this.canvasHeight = (this.props.height - 2 * pad) * config.viewScale;
      this.canvasWidth = this.canvasHeight * ratio;
      this.displayToImageRatio = this.canvasHeight / image.height;
    } else {
      this.canvasWidth = (this.props.width - 2 * pad) * config.viewScale;
      this.canvasHeight = this.canvasWidth / ratio;
      this.displayToImageRatio = this.canvasWidth / image.width;
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
    this.canvas.style.right = 'auto';
    this.canvas.style.bottom = 'auto';

    this.scale = config.viewScale;
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  render() {
    const {classes} = this.props;
    return (<canvas className={classes.canvas} ref={(canvas) => {
      if (canvas) {
        this.canvas = canvas;
        this.context = canvas.getContext('2d');
        if (this.props.width && this.props.height &&
            getCurrentItem().loaded) {
          this.updateScale();
        }
      }
    }}/>);
  }

  /**
   * Execute when component state is updated
   */
  componentDidUpdate() {
    this.redraw();
  }

  /**
   * Handles canvas redraw
   * @return {boolean}
   */
  redraw(): boolean {
    // TODO: should support lazy drawing
    let state = Session.getState();
    let item = state.current.item;
    let loaded = state.items[item].loaded;
    if (loaded) {
      let image = Session.images[item];
      // draw stuff
      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.context.drawImage(image, 0, 0, image.width, image.height,
        0, 0, this.canvas.width, this.canvas.height);
    }
    return true;
  }
}

export default withStyles(styles, {withTheme: true})(ImageView);
