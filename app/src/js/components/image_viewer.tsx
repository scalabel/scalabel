import { withStyles } from '@material-ui/core/styles'
import * as React from 'react'
import Session from '../common/session'
import { getCurrentViewerConfig, isCurrentItemLoaded } from '../functional/state_util'
import { ImageViewerConfigType, State } from '../functional/types'
import { imageViewStyle } from '../styles/label'
import {
  drawImageOnCanvas,
  MAX_SCALE,
  MIN_SCALE,
  updateCanvasScale
} from '../view_config/image'
import { Viewer } from './viewer'

interface ClassType {
  /** image canvas */
  image_canvas: string
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
export class ImageViewer extends Viewer<Props> {
  /** The image context */
  public imageContext: CanvasRenderingContext2D | null

  /** The image canvas */
  private imageCanvas: HTMLCanvasElement | null
  /** The mask to hold the display */
  private display: HTMLDivElement | null

  // display variables
  /** The current scale */
  private scale: number

  /**
   * Constructor, handles subscription to store
   * @param {Object} props: react props
   */
  constructor (props: Readonly<Props>) {
    super(props)

    // constants

    // initialization
    this.scale = 1
    this.imageContext = null
    this.imageCanvas = null
    this.display = null
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render () {
    const { classes } = this.props
    let imageCanvas = (<canvas
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
            && this.currentItemIsLoaded()
            && this.imageContext) {
            this.updateScale(this.imageCanvas, this.imageContext, true)
          }
        }
      }}
    />)

    if (this.display) {
      const displayRect = this.display.getBoundingClientRect()
      imageCanvas = React.cloneElement(
        imageCanvas,
        { height: displayRect.height, width: displayRect.width }
      )
    }

    return imageCanvas
  }

  /**
   * Function to redraw all canvases
   * @return {boolean}
   */
  public redraw (): boolean {
    if (this.currentItemIsLoaded() && this.imageCanvas && this.imageContext) {
      const item = this.state.user.select.item
      const sensor = this.state.user.viewerConfigs[this.props.id].sensor
      if (item < Session.images.length &&
          sensor in Session.images[item]) {
        const image = Session.images[item][sensor]
        // redraw imageCanvas
        drawImageOnCanvas(this.imageCanvas, this.imageContext, image)
      }
    }
    return true
  }

  /**
   * notify state is updated
   */
  protected updateState (_state: State): void {
    if (this.display !== this.props.display) {
      this.display = this.props.display
      this.forceUpdate()
    }
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

  /**
   * function to check if the current item is loaded
   * @return {boolean}
   */
  private currentItemIsLoaded (): boolean {
    return isCurrentItemLoaded(this.state)
  }
}

export default withStyles(imageViewStyle, { withTheme: true })(ImageViewer)
