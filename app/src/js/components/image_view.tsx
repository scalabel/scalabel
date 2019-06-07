import React from 'react';
import Session from '../common/session';
import {ImageViewerConfigType} from '../functional/types';
import {withStyles} from '@material-ui/core/styles/index';
import createStyles from '@material-ui/core/styles/createStyles';

const pad = 10;

const styles: any = (theme: any) => createStyles({
  canvas: {
    position: 'relative',
  },
});

interface Props {
  classes: any;
  theme: any;
  height: number;
  width: number;
}

/**
 * Get the current item in the state
 * @return {ItemType}
 */
function getCurrentItem() {
  const state = Session.getState();
  return state.items[state.current.item];
}

/**
 * Retrieve the current viewer configuration
 * @return {ViewerConfigType}
 */
function getCurrentViewerConfig() {
  const state = Session.getState();
  return state.items[state.current.item].viewerConfig;
}

/**
 * Canvas Viewer
 */
class ImageView extends React.Component<Props> {
  private canvas: any;
  public context: any;
  private MAX_SCALE: number;
  private MIN_SCALE: number;
  // private SCALE_RATIO: number;
  private UP_RES_RATIO: number;
  private scale: number;
  // need these two below to prevent jitters caused by round off
  private canvasHeight: number;
  private canvasWidth: number;
  private displayToImageRatio: number;
  // False for image canvas, true for anything else
  private upRes: boolean;
  /**
   * Constructor, handles subscription to store
   * @param {Object} props: react props
   */
  constructor(props: any) {
    super(props);

    this.MAX_SCALE = 3.0;
    this.MIN_SCALE = 1.0;
    // this.SCALE_RATIO = 1.05;
    this.UP_RES_RATIO = 2;
    this.scale = 1;
    this.canvasHeight = props.height;
    this.canvasWidth = props.width;
    this.displayToImageRatio = 1;

    this.upRes = true;
  }

  /**
   * Convert image coordinate to canvas coordinate.
   * If affine, assumes values to be [x, y]. Otherwise
   * performs linear transformation.
   * @param {Array<number>} values - the values to convert.
   * @return {Array<number>} - the converted values.
   */
  public toCanvasCoords(values: number[]) {
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
  public toImageCoords(values: number[]) {
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
  private _getPadding() {
    return {
      x: Math.max(pad, (this.props.width - this.canvasWidth) / 2),
      y: Math.max(pad, (this.props.height - this.canvasHeight) / 2),
    };
  }

  /**
   * Set the scale of the image in the display
   */
  private updateScale() {
    const config: ImageViewerConfigType =
      getCurrentViewerConfig() as ImageViewerConfigType;

    // set scale
    if (config.viewScale >= this.MIN_SCALE
      && config.viewScale < this.MAX_SCALE) {
      const ratio = config.viewScale / this.scale;
      this.context.scale(ratio, ratio);
    } else {
      return;
    }

    // resize canvas
    const item = getCurrentItem();
    const image = Session.images[item.index];
    const ratio = (image.width + 2 * pad) / (image.height + 2 * pad);

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
    const padding = this._getPadding();
    const padX = padding.x;
    const padY = padding.y;

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
  public render() {
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
  public componentDidUpdate() {
    this.redraw();
  }

  /**
   * Handles canvas redraw
   * @return {boolean}
   */
  public redraw(): boolean {
    // TODO: should support lazy drawing
    const state = Session.getState();
    const item = state.current.item;
    const loaded = state.items[item].loaded;
    if (loaded) {
      const image = Session.images[item];
      // draw stuff
      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.context.drawImage(image, 0, 0, image.width, image.height,
        0, 0, this.canvas.width, this.canvas.height);
    }
    return true;
  }
}

export default withStyles(styles, {withTheme: true})(ImageView);
