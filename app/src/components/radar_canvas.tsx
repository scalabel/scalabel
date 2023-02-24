import { withStyles } from "@material-ui/core/styles"
import * as React from "react"
import { connect } from "react-redux"

import Session from "../common/session"
import { Key } from "../const/common"
import { getCurrentViewerConfig, isFrameLoaded } from "../functional/state_util"
import { Vector2D } from "../math/vector2d"
import { label2dViewStyle } from "../styles/label"
import { ImageViewerConfigType } from "../types/state"
import {
  clearCanvas,
  MAX_SCALE,
  MIN_SCALE,
  normalizeMouseCoordinates,
  updateCanvasScale
} from "../view_config/image"
import { Crosshair, Crosshair2D } from "./crosshair"
import {
  DrawableCanvas,
  DrawableProps,
  mapStateToDrawableProps
} from "./viewer"
import { isKeyFrame } from "./util"
import { alert } from "../common/alert"
import { Severity } from "../types/common"
import { changeRadarStatus } from "../action/common"

interface ClassType {
  /** label canvas */
  label2d_canvas: string
  /** control canvas */
  control_canvas: string
}

interface Props extends DrawableProps {
  /** styles */
  classes: ClassType
  /** display */
  display: HTMLDivElement | null
  /** viewer id */
  id: number
}

/**
 * Canvas Viewer
 */
export class RadarCanvas extends DrawableCanvas<Props> {
  /** The label context */
  public labelContext: CanvasRenderingContext2D | null
  /** The control context */
  public controlContext: CanvasRenderingContext2D | null


  /** The label canvas */
  private labelCanvas: HTMLCanvasElement | null
  /** The control canvas */
  private controlCanvas: HTMLCanvasElement | null
  /** The mask to hold the display */
  private display: HTMLDivElement | null

  // Display variables
  /** The current scale */
  private scale: number
  /** The canvas height */
  private canvasHeight: number
  /** The canvas width */
  private canvasWidth: number
  /** The scale between the display and image data */
  private displayToImageRatio: number
  /** The crosshair */
  private readonly crosshair: React.RefObject<Crosshair2D>
  /** key up listener */
  private readonly _keyUpListener: (e: KeyboardEvent) => void
  /** key down listener */
  private readonly _keyDownListener: (e: KeyboardEvent) => void

  // Keyboard and mouse status
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }

  /**
   * Constructor, handles subscription to store
   *
   * @param {Object} props: react props
   * @param props
   */
  constructor(props: Readonly<Props>) {
    super(props)

    // Constants

    // Initialization
    this._keyDownMap = {}
    this.scale = 1
    this.canvasHeight = 0
    this.canvasWidth = 0
    this.displayToImageRatio = 1
    this.controlContext = null
    this.controlCanvas = null
    this.labelContext = null
    this.labelCanvas = null
    this.display = this.props.display
    
    
    this.crosshair = React.createRef()

    this._keyUpListener = (e) => {
      this.onKeyUp(e)
    }
    this._keyDownListener = (e) => {
      this.onKeyDown(e)
    }
  }

  /**
   * Component mount callback
   */
  public componentDidMount(): void {
    super.componentDidMount()
    document.addEventListener("keydown", this._keyDownListener)
    document.addEventListener("keyup", this._keyUpListener)

  }

  /**
   * Unmount callback
   */
  public componentWillUnmount(): void {
    super.componentWillUnmount()
    document.removeEventListener("keydown", this._keyDownListener)
    document.removeEventListener("keyup", this._keyUpListener)

  }

  /**
   * Set the current cursor
   *
   * @param {string} cursor - cursor type
   */
  public setCursor(cursor: string): void {
    if (this.labelCanvas !== null) {
      this.labelCanvas.style.cursor = cursor
    }
  }

  /**
   * Set the current cursor to default
   */
  public setDefaultCursor(): void {
    this.setCursor("crosshair")
  }

  /**
   * Render function
   *
   * @return {React.Fragment} React fragment
   */
  public render(): JSX.Element[] {
    const { classes } = this.props
    let controlCanvas = (
      <canvas
        key="control-canvas"
        className={classes.control_canvas}
        ref={(canvas) => {
          if (canvas !== null) {
            this.updateCanvas(canvas, true)
          }
        }}
      />
    )
    
    let labelCanvas = (
      <canvas
        key="label2d-canvas"
        className={classes.label2d_canvas}
        ref={(canvas) => {
          if (canvas !== null) {
            this.updateCanvas(canvas, false)
          }
        }}
        onMouseDown={(e) => {
          this.onMouseDown(e)
        }}
        onMouseUp={(e) => {
          this.onMouseUp(e)
        }}
        onMouseMove={(e) => {
          this.onMouseMove(e)
        }}
      />
    )
    const ch = (
      <Crosshair
        key="crosshair-canvas"
        display={this.display}
        innerRef={this.crosshair}
      />
    )
    if (this.display !== null) {
      const displayRect = this.display.getBoundingClientRect()
      controlCanvas = React.cloneElement(controlCanvas, {
        height: displayRect.height,
        width: displayRect.width
      })
      labelCanvas = React.cloneElement(labelCanvas, {
        height: displayRect.height,
        width: displayRect.width
      })
    }

    return [ch, controlCanvas, labelCanvas]
  }

  /**
   * Function to redraw all canvases
   *
   * @return {boolean}
   */
  public redraw(): boolean {
    this.clear()
    if (
      this.labelCanvas !== null &&
      this.labelContext !== null &&
      this.controlCanvas !== null &&
      this.controlContext !== null
    ) {
      
    }
    return true
  }

  /**
   * Clear canvas
   */
  public clear(): void {
    if (
      this.labelCanvas !== null &&
      this.labelContext !== null &&
      this.controlCanvas !== null &&
      this.controlContext !== null
    ) {
      clearCanvas(this.labelCanvas, this.labelContext)
      clearCanvas(this.controlCanvas, this.controlContext)
    }
  }

  /**
   * Callback function when mouse is down
   *
   * @param {MouseEvent} e - event
   */
  public onMouseDown(e: React.MouseEvent<HTMLCanvasElement>): void {
    if (e.button !== 0 || this.checkFreeze()) {
      return
    }
    if (
      !isKeyFrame(
        this.state.user.select.item,
        this.state.task.config.keyInterval
      )
    ) {
      alert(
        Severity.WARNING,
        "You can not edit label in non-keyframe, please press CTRL + left/right arrow to the nearest keyframe"
      )
      return
    }

    // get mouse position in image coordinates
    // dispatch mouse position to state
    const mousePos = this.getMousePos(e)
    if (mousePos === null) {
      return
    }
    const item = this.state.user.select.item
    const sensor = this.state.user.viewerConfigs[this.props.id].sensor
    let pixel 
    let width
    let height
    if (
      isFrameLoaded(this.state, item, sensor) &&
      item < Session.images.length &&
      sensor in Session.images[item]
    ) {
      const image = Session.images[item][sensor]
      let canvas = document.createElement('canvas');
      canvas.width = image.width;
      canvas.height = image.height;
      const context_tmp = canvas.getContext('2d');
      if (context_tmp === null) {
        return
      }
      context_tmp.drawImage(image, 0, 0);
      pixel = context_tmp.getImageData(Math.floor(mousePos[0]), Math.floor(mousePos[1]), 1, 1).data
      width = image.width
      height = image.height  
    }
    if (pixel === undefined || width === undefined || height === undefined) {
      return
    }
    // UPDATE RADAR SCALE HERE--------------------------------------------------
    let radar_scale = 0.0432 // 0.0432m/pixel
    // -------------------------------------------------------------------------
    let raw_pos_x = Math.floor(mousePos[0])
    let raw_pos_y = Math.floor(mousePos[1])
    let world_pos_x = (width/2-raw_pos_x) * radar_scale
    let world_pos_y = (height/2-raw_pos_y) * radar_scale
    console.log([world_pos_x, world_pos_y,pixel[0]])
    Session.dispatch(changeRadarStatus([world_pos_x, world_pos_y,pixel[0]]))
  }

  /**
   * Callback function when mouse is up
   *
   * @param {MouseEvent} e - event
   */
  public onMouseUp(e: React.MouseEvent<HTMLCanvasElement>): void {
    if (e.button !== 0 || this.checkFreeze()) {
      return
    }
  }

  /**
   * Callback function when mouse moves
   *
   * @param {MouseEvent} e - event
   */
  public onMouseMove(e: React.MouseEvent<HTMLCanvasElement>): void {
    if (this.checkFreeze()) {
      return
    }

    if (this.crosshair.current !== null) {
      this.crosshair.current.onMouseMove(e)
    }




    this.setDefaultCursor()

  }

  /**
   * Callback function when key is down
   *
   * @param {KeyboardEvent} e - event
   */
  public onKeyDown(e: KeyboardEvent): void {
    if (this.checkFreeze()) {
      return
    }

    const key = e.key
    this._keyDownMap[key] = true

  }

  /**
   * Callback function when key is up
   *
   * @param {KeyboardEvent} e - event
   */
  public onKeyUp(e: KeyboardEvent): void {
    if (this.checkFreeze()) {
      return
    }

    const key = e.key
    // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
    delete this._keyDownMap[key]
    if (key === Key.CONTROL || key === Key.META) {
      // Control or command
      this.setDefaultCursor()
    }

  }

  /**
   * notify state is updated
   *
   * @param state
   */
  public updateState(): void {
    if (this.display !== this.props.display) {
      this.display = this.props.display
      this.forceUpdate()
    }

  }

  /**
   * Get the mouse position on the canvas in the image coordinates.
   *
   * @param {MouseEvent | WheelEvent} e: mouse event
   * @param e
   * @return {Vector2D}
   * mouse position (x,y) on the canvas
   */
  private getMousePos(e: React.MouseEvent<HTMLCanvasElement>): Vector2D {
    if (this.display !== null && this.labelCanvas !== null) {
      return normalizeMouseCoordinates(
        this.labelCanvas,
        this.canvasWidth,
        this.canvasHeight,
        this.displayToImageRatio,
        e.clientX,
        e.clientY
      )
    }
    return new Vector2D(0, 0)
  }

  /**
   * Update the canvas dimentions from the htmlcanvas element
   *
   * @param canva
   * @param canvas
   * @param isContorl
   */
  private updateCanvas(canvas: HTMLCanvasElement, isContorl: boolean): void {
    const context = canvas.getContext("2d")
    if (context === null) {
      return
    }
    if (canvas !== null && this.display !== null) {
      if (isContorl) {
        this.controlCanvas = canvas
        this.controlContext = context
      } else {
        this.labelCanvas = canvas
        this.labelContext = context
      }
      const displayRect = this.display.getBoundingClientRect()
      const item = this.state.user.select.item
      const sensor = this.state.user.viewerConfigs[this.props.id].sensor
      if (
        isFrameLoaded(this.state, item, sensor) &&
        displayRect.width !== 0 &&
        !isNaN(displayRect.width) &&
        displayRect.height !== 0 &&
        !isNaN(displayRect.height)
      ) {
        this.updateScale(canvas, context, true)
      }
    }
  }

  /**
   * Set the scale of the image in the display
   *
   * @param {object} canvas
   * @param context
   * @param {boolean} upRes
   */
  private updateScale(
    canvas: HTMLCanvasElement,
    context: CanvasRenderingContext2D,
    upRes: boolean
  ): void {
    if (this.display === null) {
      return
    }
    const imgConfig = getCurrentViewerConfig(
      this.state,
      this.props.id
    ) as ImageViewerConfigType
    if (imgConfig.viewScale >= MIN_SCALE && imgConfig.viewScale < MAX_SCALE) {
      ;[
        this.canvasWidth,
        this.canvasHeight,
        this.displayToImageRatio,
        this.scale
      ] = updateCanvasScale(
        this.state,
        this.display,
        canvas,
        context,
        imgConfig,
        imgConfig.viewScale / this.scale,
        upRes
      )
    }
  }
}

const styledCanvas = withStyles(label2dViewStyle, { withTheme: true })(
  RadarCanvas
)
export default connect(mapStateToDrawableProps)(styledCanvas)
