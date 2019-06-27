import { Canvas2d } from './canvas2d';
import * as React from 'react';
import Session from '../common/session';
import { ImageViewerConfigType, ViewerConfigType } from '../functional/types';
import { withStyles } from '@material-ui/core/styles';
import * as types from '../action/types';
import EventListener, { withOptions } from 'react-event-listener';
import { imageViewStyle } from '../styles/label';

interface ClassType {
  /** canvas */
  canvas: string;
  /** image display area */
  mask: string;
  /** background */
  background: string;
}

interface Props {
  /** styles */
  classes: ClassType;
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
  return state.items[state.current.item].viewerConfig as ImageViewerConfigType;
}

/**
 * Canvas Viewer
 */
class ImageView extends Canvas2d<Props> {
  /** The image canvas */
  private imageCanvas: any;
  /** The context */
  public context: any;
  /** The maximum scale */
  private readonly MAX_SCALE: number;
  /** The minimum scale */
  private readonly MIN_SCALE: number;
  /** The boosted ratio to draw shapes sharper */
  private readonly UP_RES_RATIO: number;
  /** The zoom ratio */
  private readonly ZOOM_RATIO: number;
  /** The scroll-zoom ratio */
  private readonly SCROLL_ZOOM_RATIO: number;
  /** The current scale */
  private scale: number;
  /** The canvas height */
  private canvasHeight: number;
  /** The canvas width */
  private canvasWidth: number;
  /** The scale between the display and image data */
  private displayToImageRatio: number;
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: number]: boolean };
  /** The timer for scrolling */
  private scrollTimer: number | undefined;
  /** The div to hold the display */
  private mask: any;

  /**
   * Constructor, handles subscription to store
   * @param {Object} props: react props
   */
  constructor(props: Readonly<Props>) {
    super(props);

    // constants
    this.MAX_SCALE = 3.0;
    this.MIN_SCALE = 1.0;
    this.ZOOM_RATIO = 1.05;
    this.SCROLL_ZOOM_RATIO = 1.01;
    this.UP_RES_RATIO = 2;

    // initialization
    this._keyDownMap = {};
    this.scale = 1;
    this.canvasHeight = 0;
    this.canvasWidth = 0;
    this.displayToImageRatio = 1;
    this.scrollTimer = undefined;

    // this.setupController();
    // set keyboard listeners
    document.onkeydown = this.onKeyDown.bind(this);
    document.onkeyup = this.onKeyUp.bind(this);
  }

  /**
   * Get the coordinates of the upper left corner of the image canvas
   * @return {[number]} the x and y coordinates
   */
  private getVisibleCanvasCoords() {
    const maskRect = this.mask.getBoundingClientRect();
    const imgRect = this.imageCanvas.getBoundingClientRect();
    return [maskRect.x - imgRect.x, maskRect.y - imgRect.y];
  }

  /**
   * Get the mouse position on the canvas in the image coordinates.
   * @param {MouseEvent | WheelEvent} e: mouse event
   * @return {object}:
   * mouse position (x,y) on the canvas
   */
  private getMousePos(e: MouseEvent | WheelEvent) {
    const [offsetX, offsetY] = this.getVisibleCanvasCoords();
    const maskRect = this.mask.getBoundingClientRect();
    let x = e.clientX - maskRect.x + offsetX;
    let y = e.clientY - maskRect.y + offsetY;

    // limit the mouse within the image
    x = Math.max(0, Math.min(x, this.canvasWidth));
    y = Math.max(0, Math.min(y, this.canvasHeight));

    // return in the image coordinates
    return {
      x: x / this.displayToImageRatio,
      y: y / this.displayToImageRatio
    };
  }

  /**
   * Callback function when mouse is down
   * @param {MouseEvent} _ - event
   */
  private onMouseDown(_: MouseEvent) {
    // mouse down
  }

  /**
   * Callback function when mouse is up
   * @param {MouseEvent} _ - event
   */
  private onMouseUp(_: MouseEvent) {
    // mouse up
  }

  /**
   * Callback function when mouse moves
   * @param {MouseEvent} _ - event
   */
  private onMouseMove(_: MouseEvent) {
    // mouse move
  }

  /**
   * Callback function for scrolling
   * @param {WheelEvent} e - event
   */
  private onWheel(e: WheelEvent) {
    if (this.isKeyDown('ctrl')) { // control for zoom
      e.preventDefault();
      const mousePos = this.getMousePos(e);
      if (this.scrollTimer !== undefined) {
        clearTimeout(this.scrollTimer);
      }
      if (e.deltaY < 0) {
        this.zoomHandler(this.SCROLL_ZOOM_RATIO, mousePos.x, mousePos.y);
      } else if (e.deltaY > 0) {
        this.zoomHandler(
          1 / this.SCROLL_ZOOM_RATIO, mousePos.x, mousePos.y);
      }
      this.redraw();
    }
  }

  /**
   * Callback function when double click occurs
   * @param {MouseEvent} _ - event
   */
  private onDblClick(_: MouseEvent) {
    // double click
  }

  /**
   * Callback function when key is down
   * @param {KeyboardEvent} e - event
   */
  private onKeyDown(e: KeyboardEvent) {
    const keyID = e.keyCode ? e.keyCode : e.which;
    this._keyDownMap[keyID] = true;
    if (keyID === 187) {
      // + for zooming in
      this.zoomHandler(this.ZOOM_RATIO, -1, -1);
    } else if (keyID === 189) {
      // - for zooming out
      this.zoomHandler(1 / this.ZOOM_RATIO, -1, -1);
    }
  }

  /**
   * Callback function when key is up
   * @param {KeyboardEvent} e - event
   */
  private onKeyUp(e: KeyboardEvent) {
    const keyID = e.keyCode ? e.keyCode : e.which;
    delete this._keyDownMap[keyID];
  }

  /**
   * Whether a specific key is pressed down
   * @param {string} c - the key to check
   * @return {*}
   */
  private isKeyDown(c: string) {
    if (c === 'ctrl') {
      // ctrl or command key
      return this._keyDownMap[17] || this._keyDownMap[91];
    }
    return this._keyDownMap[c.charCodeAt(0)];
  }

  /**
   * Handler for zooming
   * @param {number} zoomRatio - the zoom ratio
   * @param {number} offsetX - the offset of x for zooming to cursor
   * @param {number} offsetY - the offset of y for zooming to cursor
   */
  public zoomHandler(zoomRatio: number,
                     offsetX: number, offsetY: number) {
    const newScale = getCurrentViewerConfig().viewScale * zoomRatio;
    if (newScale >= this.MIN_SCALE && newScale <= this.MAX_SCALE) {
      Session.dispatch({
        type: types.IMAGE_ZOOM, ratio: zoomRatio,
        viewOffsetX: offsetX, viewOffsetY: offsetY
      });
    }
  }

  /**
   * Convert image coordinate to canvas coordinate.
   * If affine, assumes values to be [x, y]. Otherwise
   * performs linear transformation.
   * @param {Array<number>} values - the values to convert.
   * @param {boolean} upRes
   * @return {Array<number>} - the converted values.
   */
  public toCanvasCoords(values: number[], upRes: boolean) {
    if (values) {
      for (let i = 0; i < values.length; i++) {
        values[i] *= this.displayToImageRatio;
        if (upRes) {
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
    const maskRect = this.mask.getBoundingClientRect();
    return {
      x: Math.max(0, (maskRect.width - this.canvasWidth) / 2),
      y: Math.max(0, (maskRect.height - this.canvasHeight) / 2)
    };
  }

  /**
   * Set the scale of the image in the display
   * @param {object} canvas
   * @param {boolean} upRes
   */
  private updateScale(canvas: HTMLCanvasElement, upRes: boolean) {
    const maskRect = this.mask.getBoundingClientRect();
    const config: ViewerConfigType = getCurrentViewerConfig();
    // mouseOffset
    let mouseOffset;
    let upperLeftCoords;
    if (config.viewScale > 1.0) {
      upperLeftCoords = this.getVisibleCanvasCoords();
      if (config.viewOffsetX < 0 || config.viewOffsetY < 0) {
        mouseOffset = [
          Math.min(maskRect.width, this.imageCanvas.width) / 2,
          Math.min(maskRect.height, this.imageCanvas.height) / 2
        ];
      } else {
        mouseOffset = this.toCanvasCoords(
          [config.viewOffsetX, config.viewOffsetY], false);
        mouseOffset[0] -= upperLeftCoords[0];
        mouseOffset[1] -= upperLeftCoords[1];
      }
    }

    // set scale
    let zoomRatio;
    if (config.viewScale >= this.MIN_SCALE
      && config.viewScale < this.MAX_SCALE) {
      zoomRatio = config.viewScale / this.scale;
      this.context.scale(zoomRatio, zoomRatio);
    } else {
      return;
    }

    // resize canvas
    const item = getCurrentItem();
    const image = Session.images[item.index];
    const ratio = image.width / image.height;
    if (maskRect.width / maskRect.height > ratio) {
      this.canvasHeight = maskRect.height * config.viewScale;
      this.canvasWidth = this.canvasHeight * ratio;
      this.displayToImageRatio = this.canvasHeight
        / image.height;
    } else {
      this.canvasWidth = maskRect.width * config.viewScale;
      this.canvasHeight = this.canvasWidth / ratio;
      this.displayToImageRatio = this.canvasWidth / image.width;
    }

    // translate back to origin
    if (mouseOffset) {
      this.mask.scrollTop = this.imageCanvas.offsetTop;
      this.mask.scrollLeft = this.imageCanvas.offsetLeft;
    }

    // set canvas resolution
    if (upRes) {
      canvas.height = this.canvasHeight * this.UP_RES_RATIO;
      canvas.width = this.canvasWidth * this.UP_RES_RATIO;
    } else {
      canvas.height = this.canvasHeight;
      canvas.width = this.canvasWidth;
    }

    // set canvas size
    canvas.style.height = this.canvasHeight + 'px';
    canvas.style.width = this.canvasWidth + 'px';

    // set padding
    const padding = this._getPadding();
    const padX = padding.x;
    const padY = padding.y;

    canvas.style.left = padX + 'px';
    canvas.style.top = padY + 'px';
    canvas.style.right = 'auto';
    canvas.style.bottom = 'auto';

    // zoom to point
    if (mouseOffset && upperLeftCoords) {
      if (this.canvasWidth > maskRect.width) {
        this.mask.scrollLeft =
          zoomRatio * (upperLeftCoords[0] + mouseOffset[0])
          - mouseOffset[0];
      }
      if (this.canvasHeight > maskRect.height) {
        this.mask.scrollTop =
          zoomRatio * (upperLeftCoords[1] + mouseOffset[1])
          - mouseOffset[1];
      }
    }

    this.scale = config.viewScale;
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render() {
    const { classes } = this.props;
    const imageCanvas = (<canvas
      key='image-canvas'
      className={classes.canvas}
      ref={(canvas) => {
        if (canvas) {
          this.imageCanvas = canvas;
          this.context = canvas.getContext('2d');
          const maskRect =
            this.mask.getBoundingClientRect();
          if (maskRect.width
            && maskRect.height
            && getCurrentItem().loaded) {
            this.updateScale(canvas, false);
          }
        }
      }}
      style={{
        position: 'absolute'
      }}
    />);
    const hiddenCanvas = (<canvas
      key='hidden-canvas'
      className={classes.canvas}
      ref={(canvas) => {
        if (canvas) {
          const maskRect =
            this.mask.getBoundingClientRect();
          if (maskRect.width
            && maskRect.height
            && getCurrentItem().loaded) {
            this.updateScale(canvas, true);
          }
        }
      }}
      style={{
        position: 'absolute'
      }}
    />);
    const labelCanvas = (<canvas
      key='label-canvas'
      className={classes.canvas}
      ref={(canvas) => {
        if (canvas) {
          const maskRect =
            this.mask.getBoundingClientRect();
          if (maskRect.width
            && maskRect.height
            && getCurrentItem().loaded) {
            this.updateScale(canvas, true);
          }
        }
      }}
      style={{
        position: 'absolute'
      }}
    />);

    let canvasesWithProps;
    if (this.mask) {
      const maskRect = this.mask.getBoundingClientRect();
      canvasesWithProps = React.Children.map(
        [imageCanvas, hiddenCanvas, labelCanvas], (canvas) => {
          return React.cloneElement(canvas,
            { height: maskRect.height, width: maskRect.width });
        }
      );
    }

    return (
      <div className={classes.background}>
        <EventListener
          target='window'
          onMouseDown={(e) => this.onMouseDown(e)}
          onMouseMove={(e) => this.onMouseMove(e)}
          onMouseUp={(e) => this.onMouseUp(e)}
          onDblClick={(e) => this.onDblClick(e)}
          onWheel={withOptions((e) => this.onWheel(e), { passive: false })}
        />
        <div ref={(element) => {
          if (element) {
            this.mask = element;
          }
        }}
          className={classes.mask}
        >
          {canvasesWithProps}
        </div>
      </div>
    );
  }

  /**
   * Handles canvas redraw
   * @return {boolean}
   */
  protected redraw(): boolean {
    // TODO: should support lazy drawing
    // TODO: draw each canvas separately for optimization
    const state = this.state.session;
    const item = state.current.item;
    const loaded = state.items[item].loaded;
    if (loaded) {
      const image = Session.images[item];
      // draw stuff
      this.context.clearRect(
        0, 0, this.imageCanvas.width, this.imageCanvas.height);
      this.context.drawImage(image, 0, 0, image.width, image.height,
        0, 0, this.imageCanvas.width, this.imageCanvas.height);
    }
    return true;
  }
}

export default withStyles(imageViewStyle, { withTheme: true })(ImageView);
