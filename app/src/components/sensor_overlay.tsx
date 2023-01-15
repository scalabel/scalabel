import { withStyles } from "@material-ui/core/styles"
import * as React from "react"
import { connect } from "react-redux"

import Session from "../common/session"
import { getCurrentViewerConfig, isFrameLoaded } from "../functional/state_util"
import { imageViewStyle } from "../styles/label"
import { ImageViewerConfigType, State } from "../types/state"
import {
  clearCanvas,
  drawImageOnCanvas,
  MAX_SCALE,
  MIN_SCALE,
  updateCanvasScale
} from "../view_config/image"
import {
  DrawableCanvas,
  DrawableProps,
  mapStateToDrawableProps
} from "./viewer"

interface ClassType {
  /** image canvas */
  image_canvas: string
}

export interface Props extends DrawableProps {
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
export class SensorOverlay extends DrawableCanvas<Props> {
  /** The image context */
  protected imageContext: CanvasRenderingContext2D | null

  /** The image canvas */
  protected imageCanvas: HTMLCanvasElement | null
  /** The mask to hold the display */
  protected display: HTMLDivElement | null

  // Display variables
  /** The current scale */
  private scale: number

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
    this.scale = 1
    this.imageContext = null
    this.imageCanvas = null
    this.display = null
  }

  /**
   * Render function
   *
   * @return {React.Fragment} React fragment
   */
  public render(): JSX.Element {
    const { classes } = this.props
    let imageCanvas = (
      <canvas
        key="image-canvas"
        className={classes.image_canvas}
        ref={(canvas) => {
          if (canvas !== null && this.display !== null) {
            this.imageCanvas = canvas
            this.imageContext = canvas.getContext("2d")
            const displayRect = this.display.getBoundingClientRect()
            const item = this.state.user.select.item
            const sensor = this.state.user.viewerConfigs[this.props.id].sensor
            if (
              displayRect.width !== 0 &&
              !isNaN(displayRect.width) &&
              displayRect.height !== 0 &&
              !isNaN(displayRect.height) &&
              isFrameLoaded(this.state, item, sensor) &&
              this.imageContext !== null
            ) {
              this.updateScale(this.imageCanvas, this.imageContext, true)
            }
          }
        }}
      />
    )

    if (this.display !== null) {
      const displayRect = this.display.getBoundingClientRect()
      imageCanvas = React.cloneElement(imageCanvas, {
        height: displayRect.height,
        width: displayRect.width
      })
    }

    return imageCanvas
  }

  /**
   * Function to redraw all canvases
   *
   * @return {boolean}
   */
  public redraw(): boolean {
    if (this.imageCanvas !== null && this.imageContext !== null) {
      const item = this.state.user.select.item
      //const sensor = this.state.user.viewerConfigs[this.props.id].sensor
      
      if (
        this.state.session.overlayStatus.length > 0 
      ) {
        
        for (const sensor of this.state.session.overlayStatus){
          if (isFrameLoaded(this.state, item, sensor) &&
              item < Session.images.length &&
              sensor in Session.images[item]){
                const image = Session.images[item][sensor]
                // Redraw imageCanvas
                drawImageOnCanvas(this.imageCanvas, this.imageContext, image)
          } 
          //TODO: perhaps clear canvas necessary here 
          
        }
        
      } else {
        clearCanvas(this.imageCanvas, this.imageContext)
      }
    }
    return true
  }

  /**
   * notify state is updated
   *
   * @param _state
   */
  protected updateState(_state: State): void {
    if (this.display !== this.props.display) {
      this.display = this.props.display
      this.forceUpdate()
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
      const newParams = updateCanvasScale(
        this.state,
        this.display,
        canvas,
        context,
        imgConfig,
        imgConfig.viewScale / this.scale,
        upRes
      )
      this.scale = newParams[3]
    }
  }
}

const styledCanvas = withStyles(imageViewStyle, { withTheme: true })(
  SensorOverlay
)
export default connect(mapStateToDrawableProps)(styledCanvas)
