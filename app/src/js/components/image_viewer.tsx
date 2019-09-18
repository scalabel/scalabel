import { withStyles } from '@material-ui/core/styles'
import * as React from 'react'
import Session from '../common/session'
import { getCurrentImageViewerConfig, isItemLoaded } from '../functional/state_util'
import { State } from '../functional/types'
import { imageViewStyle } from '../styles/label'
import {
  drawImageOnCanvas,
  MAX_SCALE,
  MIN_SCALE,
  updateCanvasScale
} from '../view/image'
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
      const image = Session.images[this.state.session.user.select.item]
      // redraw imageCanvas
      drawImageOnCanvas(this.imageCanvas, this.imageContext, image)
    }
    return true
  }

  /**
   * notify state is updated
   */
  protected updateState (_state: State): void {
    this.display = this.props.display
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
    const state = Session.getState()
    const config =
      getCurrentImageViewerConfig(state)

    if (config.viewScale < MIN_SCALE || config.viewScale >= MAX_SCALE) {
      return
    }
    const newParams = updateCanvasScale(
      this.display,
      canvas,
      context,
      config,
      config.viewScale / this.scale,
      upRes
    )
    this.scale = newParams[3]
  }

  /**
   * function to check if the current item is loaded
   * @return {boolean}
   */
  private currentItemIsLoaded (): boolean {
    return isItemLoaded(this.state.session)
  }
}

export default withStyles(imageViewStyle, { withTheme: true })(ImageViewer)
