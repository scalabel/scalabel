import { withStyles } from '@material-ui/core/styles'
import * as React from 'react'
import EventListener, { withOptions } from 'react-event-listener'
import { zoomImage } from '../action/image'
import Session from '../common/session'
import { Label2DList } from '../drawable/label2d_list'
import { decodeControlIndex, rgbToIndex } from '../drawable/util'
import { ImageViewerConfigType, ItemType, State, ViewerConfigType } from '../functional/types'
import { Size2D } from '../math/size2d'
import { Vector2D } from '../math/vector2d'
import { imageViewStyle } from '../styles/label'
import { Canvas2d } from './canvas2d'

interface ClassType {
  /** image canvas */
  image_canvas: string
  /** label canvas */
  label_canvas: string
  /** control canvas */
  control_canvas: string
  /** image display area */
  display: string
  /** background */
  background: string
}

interface Props {
  /** styles */
  classes: ClassType
}

/**
 * Get the current item in the state
 * @return {ItemType}
 */
function getCurrentItem (): ItemType {
  const state = Session.getState()
  return state.items[state.current.item]
}

/**
 * Retrieve the current viewer configuration
 * @return {ImageViewerConfigType}
 */
function getCurrentViewerConfig (): ImageViewerConfigType {
  const state = Session.getState()
  return state.items[state.current.item].viewerConfig as ImageViewerConfigType
}

/**
 * Function to find mode of a number array.
 * @param {number[]} arr - the array.
 * @return {number} the mode of the array.
 */
export function mode (arr: number[]) {
  return arr.sort((a, b) =>
    arr.filter((v) => v === a).length
    - arr.filter((v) => v === b).length
  ).pop()
}

/**
 * Canvas Viewer
 */
export class ImageView extends Canvas2d<Props> {
  /** The image context */
  public imageContext: CanvasRenderingContext2D | null
  /** The label context */
  public labelContext: CanvasRenderingContext2D | null
  /** The control context */
  public controlContext: CanvasRenderingContext2D | null

  /** drawable label list */
  private _labels: Label2DList
  /** The image canvas */
  private imageCanvas: HTMLCanvasElement | null
  /** The label canvas */
  private labelCanvas: HTMLCanvasElement | null
  /** The control canvas */
  private controlCanvas: HTMLCanvasElement | null
  /** The mask to hold the display */
  private display: HTMLDivElement | null

  // display constants
  /** The maximum scale */
  private readonly MAX_SCALE: number
  /** The minimum scale */
  private readonly MIN_SCALE: number
  /** The boosted ratio to draw shapes sharper */
  private readonly UP_RES_RATIO: number
  /** The zoom ratio */
  private readonly ZOOM_RATIO: number
  /** The scroll-zoom ratio */
  private readonly SCROLL_ZOOM_RATIO: number

  // display variables
  /** The current scale */
  private scale: number
  /** The canvas height */
  private canvasHeight: number
  /** The canvas width */
  private canvasWidth: number
  /** The scale between the display and image data */
  private displayToImageRatio: number

  // keyboard and mouse status
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }

  // scrolling
  /** The timer for scrolling */
  private scrollTimer: number | undefined

  // grabbing
  /** Whether or not the mouse is currently grabbing the image */
  private _isGrabbingImage: boolean
  /** The x coordinate when the grab starts */
  private _startGrabX: number
  /** The y coordinate when the grab starts */
  private _startGrabY: number
  /** The visible coordinates when the grab starts */
  private _startGrabVisibleCoords: number[]

  /**
   * Constructor, handles subscription to store
   * @param {Object} props: react props
   */
  constructor (props: Readonly<Props>) {
    super(props)

    // constants
    this.MAX_SCALE = 3.0
    this.MIN_SCALE = 1.0
    this.ZOOM_RATIO = 1.05
    this.SCROLL_ZOOM_RATIO = 1.01
    this.UP_RES_RATIO = 2

    // initialization
    this._keyDownMap = {}
    this._isGrabbingImage = false
    this._startGrabX = -1
    this._startGrabY = -1
    this._startGrabVisibleCoords = []
    this.scale = 1
    this.canvasHeight = 0
    this.canvasWidth = 0
    this.displayToImageRatio = 1
    this.scrollTimer = undefined
    this.imageContext = null
    this.imageCanvas = null
    this.controlContext = null
    this.controlCanvas = null
    this.labelContext = null
    this.labelCanvas = null
    this.display = null

    // set keyboard listeners
    document.onkeydown = this.onKeyDown.bind(this)
    document.onkeyup = this.onKeyUp.bind(this)

    this._labels = new Label2DList()
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
   * Get the current item in the state
   * @return {Size2D}
   */
  public getCurrentImageSize (): Size2D {
    const item = getCurrentItem()
    const image = Session.images[item.index]
    return new Size2D(image.width, image.height)
  }

  /**
   * Handler for zooming
   * @param {number} zoomRatio - the zoom ratio
   * @param {number} offsetX - the offset of x for zooming to cursor
   * @param {number} offsetY - the offset of y for zooming to cursor
   */
  public zoomHandler (zoomRatio: number,
                      offsetX: number, offsetY: number) {
    const newScale = getCurrentViewerConfig().viewScale * zoomRatio
    if (newScale >= this.MIN_SCALE && newScale <= this.MAX_SCALE) {
      Session.dispatch(zoomImage(zoomRatio, offsetX, offsetY))
    }
  }

  /**
   * Convert image coordinate to canvas coordinate.
   * If affine, assumes values to be [x, y]. Otherwise
   * performs linear transformation.
   * @param {Vector2D} values - the values to convert.
   * @param {boolean} upRes
   * @return {Vector2D} - the converted values.
   */
  public toCanvasCoords (values: Vector2D, upRes: boolean): Vector2D {
    const out = values.clone().scale(this.displayToImageRatio)
    if (upRes) {
      out.scale(this.UP_RES_RATIO)
    }
    return out
  }

  /**
   * Convert canvas coordinate to image coordinate.
   * If affine, assumes values to be [x, y]. Otherwise
   * performs linear transformation.
   * @param {Vector2D} values - the values to convert.
   * @param {boolean} upRes - whether the canvas has higher resolution
   * @return {Vector2D} - the converted values.
   */
  public toImageCoords (values: Vector2D, upRes: boolean = true): Vector2D {
    const up = (upRes) ? 1 / this.UP_RES_RATIO : 1
    return values.clone().scale(this.displayToImageRatio * up)
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render () {
    const { classes } = this.props
    const controlCanvas = (<canvas
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
            && getCurrentItem().loaded) {
            this.updateScale(canvas, true)
          }
        }
      }}
    />)
    const labelCanvas = (<canvas
      key='label-canvas'
      className={classes.label_canvas}
      ref={(canvas) => {
        if (canvas && this.display) {
          this.labelCanvas = canvas
          this.labelContext = canvas.getContext('2d')
          const displayRect =
            this.display.getBoundingClientRect()
          if (displayRect.width
            && displayRect.height
            && getCurrentItem().loaded) {
            this.updateScale(canvas, true)
          }
        }
      }}
    />)
    const imageCanvas = (<canvas
      key='image-canvas'
      className={classes.image_canvas}
      ref={(canvas) => {
        if (canvas && this.display) {
          this.imageCanvas = canvas
          this.imageContext = canvas.getContext('2d')
          const displayRect =
            this.display.getBoundingClientRect()
          if (displayRect.width
            && displayRect.height
            && getCurrentItem().loaded) {
            this.updateScale(canvas, false)
          }
        }
      }}
    />)

    let canvasesWithProps
    if (this.display) {
      const displayRect = this.display.getBoundingClientRect()
      canvasesWithProps = React.Children.map(
        [imageCanvas, controlCanvas, labelCanvas], (canvas) => {
          return React.cloneElement(canvas,
            { height: displayRect.height, width: displayRect.width })
        }
      )
    }

    return (
      <div className={classes.background}>
        <EventListener
          target='window'
          onMouseDown={(e) => this.onMouseDown(e)}
          onMouseMove={(e) => this.onMouseMove(e)}
          onMouseUp={(e) => this.onMouseUp(e)}
          onMouseLeave={(e) => this.onMouseLeave(e)}
          onDblClick={(e) => this.onDblClick(e)}
          onWheel={withOptions((e) => this.onWheel(e), { passive: false })}
        />
        <div ref={(element) => {
          if (element) {
            this.display = element
          }
        }}
          className={classes.display}
        >
          {canvasesWithProps}
        </div>
      </div>
    )
  }

  /**
   * Function to redraw all canvases
   * @return {boolean}
   */
  public redraw (): boolean {
    // redraw imageCanvas
    if (this.currentItemIsLoaded()) {
      this.redrawImageCanvas()
    }
    this.redrawLabels()
    return true
  }

  /**
   * Function to redraw the image canvas
   */
  public redrawImageCanvas () {
    if (this.currentItemIsLoaded() && this.imageCanvas && this.imageContext) {
      const image = Session.images[this.state.session.current.item]
      // redraw imageCanvas
      this.clearCanvas(this.imageCanvas, this.imageContext)
      this.imageContext.drawImage(image, 0, 0, image.width, image.height,
        0, 0, this.imageCanvas.width, this.imageCanvas.height)
    }
    return true
  }

  /**
   * Redraw the labels
   */
  public redrawLabels () {
    if (this.labelCanvas !== null && this.labelContext !== null &&
      this.controlCanvas !== null && this.controlContext !== null) {
      this.clearCanvas(this.labelCanvas, this.labelContext)
      this.clearCanvas(this.controlCanvas, this.controlContext)
      this._labels.redraw(this.labelContext, this.controlContext,
        this.displayToImageRatio * this.UP_RES_RATIO)
    }
  }

  /**
   * notify state is updated
   */
  protected updateState (state: State): void {
    this._labels.updateState(state, state.current.item)
  }

  /**
   * Clear the canvas
   * @param {HTMLCanvasElement} canvas - the canvas to redraw
   * @param {any} context - the context to redraw
   * @return {boolean}
   */
  protected clearCanvas (canvas: HTMLCanvasElement,
                         context: CanvasRenderingContext2D): boolean {
    // clear context
    context.clearRect(0, 0, canvas.width, canvas.height)
    return true
  }

  /**
   * Get the coordinates of the upper left corner of the image canvas
   * @return {Vector2D} the x and y coordinates
   */
  private getVisibleCanvasCoords (): Vector2D {
    if (this.display && this.imageCanvas) {
      const displayRect = this.display.getBoundingClientRect() as DOMRect
      const imgRect = this.imageCanvas.getBoundingClientRect() as DOMRect
      return new Vector2D(displayRect.x - imgRect.x, displayRect.y - imgRect.y)
    }
    return new Vector2D(0, 0)
  }

  /**
   * Get the mouse position on the canvas in the image coordinates.
   * @param {MouseEvent | WheelEvent} e: mouse event
   * @return {Vector2D}
   * mouse position (x,y) on the canvas
   */
  private getMousePos (e: MouseEvent | WheelEvent): Vector2D {
    if (this.display) {
      const [offsetX, offsetY] = this.getVisibleCanvasCoords()
      const displayRect = this.display.getBoundingClientRect() as DOMRect
      let x = e.clientX - displayRect.x + offsetX
      let y = e.clientY - displayRect.y + offsetY

      // limit the mouse within the image
      x = Math.max(0, Math.min(x, this.canvasWidth))
      y = Math.max(0, Math.min(y, this.canvasHeight))

      // return in the image coordinates
      return new Vector2D(x / this.displayToImageRatio,
        y / this.displayToImageRatio)
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
      const [x, y] = this.toCanvasCoords(mousePos,
        true)
      const data = this.controlContext.getImageData(x, y, 4, 4).data
      const arr = []
      for (let i = 0; i < 16; i++) {
        const color = rgbToIndex(Array.from(data.slice(i * 4, i * 4 + 3)))
        arr.push(color)
      }
      // finding the mode of the data array to deal with anti-aliasing
      const hoveredIndex = mode(arr) as number
      return decodeControlIndex(hoveredIndex)
    } else {
      return [-1, 0]
    }
  }

  /**
   * Callback function when mouse is down
   * @param {MouseEvent} e - event
   */
  private onMouseDown (e: MouseEvent) {
    // ctrl + click for dragging
    if (this.isKeyDown('ctrl')) {
      if (this.display && this.imageCanvas) {
        const display = this.display.getBoundingClientRect()
        if (this.imageCanvas.width > display.width ||
          this.imageCanvas.height > display.height) {
          // if needed, start grabbing
          this.setCursor('grabbing')
          this._isGrabbingImage = true
          this._startGrabX = e.clientX
          this._startGrabY = e.clientY
          this._startGrabVisibleCoords = this.getVisibleCanvasCoords()
        }
      }
    } else {
      // get mouse position in image coordinates
      const mousePos = this.getMousePos(e)
      const [labelIndex, handleIndex] = this.fetchHandleId(mousePos)
      this._labels.onMouseDown(mousePos, labelIndex, handleIndex)
    }
    this.redrawLabels()
  }

  /**
   * Callback function when mouse is up
   * @param {MouseEvent} e - event
   */
  private onMouseUp (e: MouseEvent) {
    // get mouse position in image coordinates
    this._isGrabbingImage = false
    this._startGrabX = -1
    this._startGrabY = -1
    this._startGrabVisibleCoords = []

    const mousePos = this.getMousePos(e)
    const [labelIndex, handleIndex] = this.fetchHandleId(mousePos)
    this._labels.onMouseUp(mousePos, labelIndex, handleIndex)
    this.redrawLabels()
  }

  /**
   * Callback function when mouse leaves
   * @param {MouseEvent} e - event
   */
  private onMouseLeave (e: MouseEvent) {
    this._keyDownMap = {}
    this.onMouseUp(e)
  }

  /**
   * Callback function when mouse moves
   * @param {MouseEvent} e - event
   */
  private onMouseMove (e: MouseEvent) {
    // TODO: update hovered label
    // grabbing image
    if (this.isKeyDown('ctrl')) {
      if (this._isGrabbingImage) {
        if (this.display) {
          this.setCursor('grabbing')
          const dx = e.clientX - this._startGrabX
          const dy = e.clientY - this._startGrabY
          this.display.scrollLeft = this._startGrabVisibleCoords[0] - dx
          this.display.scrollTop = this._startGrabVisibleCoords[1] - dy
        }
      } else {
        this.setCursor('grab')
      }
    } else {
      this.setDefaultCursor()
    }

    // update the currently hovered shape
    const mousePos = this.getMousePos(e)
    const [labelIndex, handleIndex] = this.fetchHandleId(mousePos)
    this._labels.onMouseMove(
      mousePos, this.getCurrentImageSize(), labelIndex, handleIndex)
    this.redrawLabels()
  }

  /**
   * Callback function for scrolling
   * @param {WheelEvent} e - event
   */
  private onWheel (e: WheelEvent) {
    // get mouse position in image coordinates
    const mousePos = this.getMousePos(e)
    if (this.isKeyDown('ctrl')) { // control for zoom
      e.preventDefault()
      if (this.scrollTimer !== undefined) {
        clearTimeout(this.scrollTimer)
      }
      if (e.deltaY < 0) {
        this.zoomHandler(this.SCROLL_ZOOM_RATIO, mousePos[0], mousePos[1])
      } else if (e.deltaY > 0) {
        this.zoomHandler(
          1 / this.SCROLL_ZOOM_RATIO, mousePos[0], mousePos[1])
      }
      this.redraw()
    }
  }

  /**
   * Callback function when double click occurs
   * @param {MouseEvent} e - event
   */
  private onDblClick (_e: MouseEvent) {
    // get mouse position in image coordinates
    // const mousePos = this.getMousePos(e)
    // label-specific handling of double click
    // this.getCurrentController().onDblClick(mousePos)
  }

  /**
   * Callback function when key is down
   * @param {KeyboardEvent} e - event
   */
  private onKeyDown (e: KeyboardEvent) {
    const key = e.key
    this._keyDownMap[key] = true
    if (key === '+') {
      // + for zooming in
      this.zoomHandler(this.ZOOM_RATIO, -1, -1)
    } else if (key === '-') {
      // - for zooming out
      this.zoomHandler(1 / this.ZOOM_RATIO, -1, -1)
    }
  }

  /**
   * Callback function when key is up
   * @param {KeyboardEvent} e - event
   */
  private onKeyUp (e: KeyboardEvent) {
    const key = e.key
    delete this._keyDownMap[key]
    if (key === 'Ctrl' || key === 'Meta') {
      // ctrl or command
      this.setDefaultCursor()
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
   * Get the padding for the image given its size and canvas size.
   * @return {Vector2D} padding
   */
  private _getPadding (): Vector2D {
    if (this.display) {
      const displayRect = this.display.getBoundingClientRect()
      return new Vector2D(
        Math.max(0, (displayRect.width - this.canvasWidth) / 2),
        Math.max(0, (displayRect.height - this.canvasHeight) / 2))
    }
    return new Vector2D(0, 0)
  }

  /**
   * Set the scale of the image in the display
   * @param {object} canvas
   * @param {boolean} upRes
   */
  private updateScale (canvas: HTMLCanvasElement, upRes: boolean) {
    if (!this.display || !this.imageCanvas || !this.imageContext) {
      return
    }
    const displayRect = this.display.getBoundingClientRect()
    const config: ViewerConfigType = getCurrentViewerConfig()
    // mouseOffset
    let mouseOffset
    let upperLeftCoords
    if (config.viewScale > 1.0) {
      upperLeftCoords = this.getVisibleCanvasCoords()
      if (config.viewOffsetX < 0 || config.viewOffsetY < 0) {
        mouseOffset = [
          Math.min(displayRect.width, this.imageCanvas.width) / 2,
          Math.min(displayRect.height, this.imageCanvas.height) / 2
        ]
      } else {
        mouseOffset = this.toCanvasCoords(
          new Vector2D(config.viewOffsetX, config.viewOffsetY), false)
        mouseOffset[0] -= upperLeftCoords[0]
        mouseOffset[1] -= upperLeftCoords[1]
      }
    }

    // set scale
    let zoomRatio
    if (config.viewScale >= this.MIN_SCALE
      && config.viewScale < this.MAX_SCALE) {
      zoomRatio = config.viewScale / this.scale
      this.imageContext.scale(zoomRatio, zoomRatio)
    } else {
      return
    }

    // resize canvas
    const item = getCurrentItem()
    const image = Session.images[item.index]
    const ratio = image.width / image.height
    if (displayRect.width / displayRect.height > ratio) {
      this.canvasHeight = displayRect.height * config.viewScale
      this.canvasWidth = this.canvasHeight * ratio
      this.displayToImageRatio = this.canvasHeight
        / image.height
    } else {
      this.canvasWidth = displayRect.width * config.viewScale
      this.canvasHeight = this.canvasWidth / ratio
      this.displayToImageRatio = this.canvasWidth / image.width
    }

    // translate back to origin
    if (mouseOffset) {
      this.display.scrollTop = this.imageCanvas.offsetTop
      this.display.scrollLeft = this.imageCanvas.offsetLeft
    }

    // set canvas resolution
    if (upRes) {
      canvas.height = this.canvasHeight * this.UP_RES_RATIO
      canvas.width = this.canvasWidth * this.UP_RES_RATIO
    } else {
      canvas.height = this.canvasHeight
      canvas.width = this.canvasWidth
    }

    // set canvas size
    canvas.style.height = this.canvasHeight + 'px'
    canvas.style.width = this.canvasWidth + 'px'

    // set padding
    const padding = this._getPadding()
    const padX = padding.x
    const padY = padding.y

    canvas.style.left = padX + 'px'
    canvas.style.top = padY + 'px'
    canvas.style.right = 'auto'
    canvas.style.bottom = 'auto'

    // zoom to point
    if (mouseOffset && upperLeftCoords) {
      if (this.canvasWidth > displayRect.width) {
        this.display.scrollLeft =
          zoomRatio * (upperLeftCoords[0] + mouseOffset[0])
          - mouseOffset[0]
      }
      if (this.canvasHeight > displayRect.height) {
        this.display.scrollTop =
          zoomRatio * (upperLeftCoords[1] + mouseOffset[1])
          - mouseOffset[1]
      }
    }

    this.scale = config.viewScale
  }

  /**
   * function to check if the current item is loaded
   * @return {boolean}
   */
  private currentItemIsLoaded (): boolean {
    const state = this.state.session
    const item = state.current.item
    return state.items[item].loaded
  }
}

export default withStyles(imageViewStyle, { withTheme: true })(ImageView)
