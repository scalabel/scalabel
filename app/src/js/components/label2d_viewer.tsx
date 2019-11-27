import { withStyles } from '@material-ui/core/styles'
import * as React from 'react'
import Session from '../common/session'
import { Key } from '../common/types'
import { Label2DHandler } from '../drawable/2d/label2d_handler'
import { getCurrentViewerConfig } from '../functional/state_util'
import { ImageViewerConfigType, State } from '../functional/types'
import { Vector2D } from '../math/vector2d'
import { label2dViewStyle } from '../styles/label'
import {
  clearCanvas,
  getCurrentImageSize,
  imageDataToHandleId,
  MAX_SCALE,
  MIN_SCALE,
  normalizeMouseCoordinates,
  toCanvasCoords,
  UP_RES_RATIO,
  updateCanvasScale
} from '../view_config/image'
import { Crosshair, Crosshair2D } from './crosshair'
import { Viewer } from './viewer'

interface ClassType {
  /** label canvas */
  label2d_canvas: string
  /** control canvas */
  control_canvas: string
}

interface Props {
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
export class Label2dViewer extends Viewer<Props> {
  /** The label context */
  public labelContext: CanvasRenderingContext2D | null
  /** The control context */
  public controlContext: CanvasRenderingContext2D | null

  /** drawable label list */
  private _labelHandler: Label2DHandler
  /** The label canvas */
  private labelCanvas: HTMLCanvasElement | null
  /** The control canvas */
  private controlCanvas: HTMLCanvasElement | null
  /** The mask to hold the display */
  private display: HTMLDivElement | null

  // display variables
  /** The current scale */
  private scale: number
  /** The canvas height */
  private canvasHeight: number
  /** The canvas width */
  private canvasWidth: number
  /** The scale between the display and image data */
  private displayToImageRatio: number
  /** The crosshair */
  private crosshair: React.RefObject<Crosshair2D>
  /** key up listener */
  private _keyUpListener: (e: KeyboardEvent) => void
  /** key down listener */
  private _keyDownListener: (e: KeyboardEvent) => void

  // keyboard and mouse status
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }
  /** drawable callback */
  private _drawableUpdateCallback: () => void

  /**
   * Constructor, handles subscription to store
   * @param {Object} props: react props
   */
  constructor (props: Readonly<Props>) {
    super(props)

    // constants

    // initialization
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
    this._labelHandler = new Label2DHandler()
    this.crosshair = React.createRef()

    this._keyUpListener = (e) => { this.onKeyUp(e) }
    this._keyDownListener = (e) => { this.onKeyDown(e) }
    this._drawableUpdateCallback = this.redraw.bind(this)
  }

  /**
   * Component mount callback
   */
  public componentDidMount () {
    super.componentDidMount()
    document.addEventListener('keydown', this._keyDownListener)
    document.addEventListener('keyup', this._keyUpListener)
    Session.label2dList.subscribe(this._drawableUpdateCallback)
  }

  /**
   * Unmount callback
   */
  public componentWillUnmount () {
    super.componentWillUnmount()
    document.removeEventListener('keydown', this._keyDownListener)
    document.removeEventListener('keyup', this._keyUpListener)
    Session.label2dList.unsubscribe(this._drawableUpdateCallback)
  }

  /**
   * Set the current cursor
   * @param {string} cursor - cursor type
   */
  public setCursor (cursor: string) {
    if (this.labelCanvas !== null) {
      this.labelCanvas.style.cursor = cursor
    }
  }

  /**
   * Set the current cursor to default
   */
  public setDefaultCursor () {
    this.setCursor('crosshair')
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render () {
    const { classes } = this.props
    let controlCanvas = (<canvas
      key='control-canvas'
      className={classes.control_canvas}
      ref={(canvas) => {
        if (canvas && this.display) {
          this.controlCanvas = canvas
          this.controlContext = canvas.getContext('2d')
          const displayRect =
            this.display.getBoundingClientRect()
          if (displayRect.width
            && displayRect.height
            && this.controlContext) {
            this.updateScale(this.controlCanvas, this.controlContext, true)
          }
        }
      }}
    />)
    let labelCanvas = (<canvas
      key='label2d-canvas'
      className={classes.label2d_canvas}
      ref={(canvas) => {
        if (canvas && this.display) {
          this.labelCanvas = canvas
          this.labelContext = canvas.getContext('2d')
          const displayRect =
            this.display.getBoundingClientRect()
          if (displayRect.width
            && displayRect.height
            && this.labelContext) {
            this.updateScale(this.labelCanvas, this.labelContext, true)
          }
        }
      }}
      onMouseDown={(e) => { this.onMouseDown(e) }}
      onMouseUp={(e) => { this.onMouseUp(e) }}
      onMouseMove={(e) => { this.onMouseMove(e) }}
    />)
    const ch = (<Crosshair
      display={this.display}
      innerRef={this.crosshair}
      />)
    if (this.display) {
      const displayRect = this.display.getBoundingClientRect()
      controlCanvas = React.cloneElement(controlCanvas,
         { height: displayRect.height, width: displayRect.width })
      labelCanvas = React.cloneElement(labelCanvas,
         { height: displayRect.height, width: displayRect.width })
    }

    return [ch, controlCanvas, labelCanvas]
  }

  /**
   * Function to redraw all canvases
   * @return {boolean}
   */
  public redraw (): boolean {
    if (this.labelCanvas !== null && this.labelContext !== null &&
      this.controlCanvas !== null && this.controlContext !== null) {
      clearCanvas(this.labelCanvas, this.labelContext)
      clearCanvas(this.controlCanvas, this.controlContext)
      Session.label2dList.redraw(this.labelContext, this.controlContext,
        this.displayToImageRatio * UP_RES_RATIO)
    }
    return true
  }

  /**
   * Callback function when mouse is down
   * @param {MouseEvent} e - event
   */
  public onMouseDown (e: React.MouseEvent<HTMLCanvasElement>) {
    if (e.button !== 0 || this.checkFreeze()) {
      return
    }
    // Control + click for dragging
      // get mouse position in image coordinates
    const mousePos = this.getMousePos(e)
    const [labelIndex, handleIndex] = this.fetchHandleId(mousePos)
    if (!this.isKeyDown(Key.META) && !this.isKeyDown(Key.CONTROL)) {
      if (this._labelHandler.onMouseDown(mousePos, labelIndex, handleIndex)) {
        e.stopPropagation()
      }
    }

    Session.label2dList.onDrawableUpdate()
  }

  /**
   * Callback function when mouse is up
   * @param {MouseEvent} e - event
   */
  public onMouseUp (e: React.MouseEvent<HTMLCanvasElement>) {
    if (e.button !== 0 || this.checkFreeze()) {
      return
    }

    const mousePos = this.getMousePos(e)
    const [labelIndex, handleIndex] = this.fetchHandleId(mousePos)
    this._labelHandler.onMouseUp(mousePos, labelIndex, handleIndex)
    Session.label2dList.onDrawableUpdate()
  }

  /**
   * Callback function when mouse moves
   * @param {MouseEvent} e - event
   */
  public onMouseMove (e: React.MouseEvent<HTMLCanvasElement>) {
    if (this.checkFreeze()) {
      return
    }

    // TODO: update hovered label
    // grabbing image
    if (!this.isKeyDown(Key.CONTROL) && !this.isKeyDown(Key.META)) {
      this.setDefaultCursor()
    }

    if (this.crosshair.current) {
      this.crosshair.current.onMouseMove(e)
    }

    // update the currently hovered shape
    const mousePos = this.getMousePos(e)
    const [labelIndex, handleIndex] = this.fetchHandleId(mousePos)
    if (this._labelHandler.onMouseMove(
      mousePos,
      getCurrentImageSize(this.state, this.props.id),
      labelIndex, handleIndex
    )) {
      e.stopPropagation()
    }
    Session.label2dList.onDrawableUpdate()
  }

  /**
   * Callback function when key is down
   * @param {KeyboardEvent} e - event
   */
  public onKeyDown (e: KeyboardEvent) {
    if (this.checkFreeze()) {
      return
    }

    const key = e.key
    this._keyDownMap[key] = true
    this._labelHandler.onKeyDown(e)
    Session.label2dList.onDrawableUpdate()
  }

  /**
   * Callback function when key is up
   * @param {KeyboardEvent} e - event
   */
  public onKeyUp (e: KeyboardEvent) {
    if (this.checkFreeze()) {
      return
    }

    const key = e.key
    delete this._keyDownMap[key]
    if (key === Key.CONTROL || key === Key.META) {
      // Control or command
      this.setDefaultCursor()
    }
    this._labelHandler.onKeyUp(e)
    Session.label2dList.onDrawableUpdate()
  }

  /**
   * notify state is updated
   */
  protected updateState (state: State): void {
    if (this.display !== this.props.display) {
      this.display = this.props.display
      this.forceUpdate()
    }
    this._labelHandler.updateState(state)
  }

  /**
   * Get the mouse position on the canvas in the image coordinates.
   * @param {MouseEvent | WheelEvent} e: mouse event
   * @return {Vector2D}
   * mouse position (x,y) on the canvas
   */
  private getMousePos (e: React.MouseEvent<HTMLCanvasElement>): Vector2D {
    if (this.display && this.labelCanvas) {
      return normalizeMouseCoordinates(
        this.display,
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
   * Get the label under the mouse.
   * @param {Vector2D} mousePos: position of the mouse
   * @return {number[]}
   */
  private fetchHandleId (mousePos: Vector2D): number[] {
    if (this.controlContext) {
      const [x, y] = toCanvasCoords(mousePos, true, this.displayToImageRatio)
      const data = this.controlContext.getImageData(x, y, 4, 4).data
      return imageDataToHandleId(data)
    } else {
      return [-1, 0]
    }
  }

  /**
   * Whether a specific key is pressed down
   * @param {string} key - the key to check
   * @return {boolean}
   */
  private isKeyDown (key: string): boolean {
    return this._keyDownMap[key]
  }

  /**
   * Set the scale of the image in the display
   * @param {object} canvas
   * @param {boolean} upRes
   */
  private updateScale (
    canvas: HTMLCanvasElement,
    context: CanvasRenderingContext2D,
    upRes: boolean
  ) {
    if (!this.display) {
      return
    }
    const imgConfig =
      getCurrentViewerConfig(this.state, this.props.id) as ImageViewerConfigType
    if (imgConfig.viewScale >= MIN_SCALE && imgConfig.viewScale < MAX_SCALE) {
      [
        this.canvasWidth,
        this.canvasHeight,
        this.displayToImageRatio,
        this.scale
      ] =
      updateCanvasScale(
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

export default withStyles(label2dViewStyle, { withTheme: true })(Label2dViewer)
