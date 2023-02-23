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

function matrixMultiplication(a: number[][], b: number[][]): number[][] {
  let result: number[][] = [];
  for (let i = 0; i < a.length; i++) {
    result[i] = [];
    for (let j = 0; j < b[0].length; j++) {
      let sum = 0;
      for (let k = 0; k < a[0].length; k++) {
        sum += a[i][k] * b[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;

  
}
function transpose(matrix: number[][]): number[][] {
  let transposedMatrix: number[][] = [];
  for (let i = 0; i < matrix[0].length; i++) {
    transposedMatrix[i] = [];
    for (let j = 0; j < matrix.length; j++) {
      transposedMatrix[i][j] = matrix[j][i];
    }
  }
  return transposedMatrix;
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
        
        let first = true
        let transparancy = this.state.session.overlayTransparency
        for (const sensor of this.state.session.overlayStatus){
          if (isFrameLoaded(this.state, item, sensor) &&
              item < Session.images.length &&
              sensor in Session.images[item]){
              const image = Session.images[item][sensor]
              // Redraw imageCanvas
              if (first){
                drawImageOnCanvas(this.imageCanvas, this.imageContext, image,false,transparancy)
                first = false
              } else {
                drawImageOnCanvas(this.imageCanvas, this.imageContext, image,true,transparancy)
              }
          } else if (sensor === 9){
            // Project a bar on the image base on the radarstatus
            const image = Session.images[item][0]
            const radarStatus = this.state.session.radarStatus
            if (radarStatus.length > 0){
              const radarWidth = this.imageCanvas.width
              const radarHeight = this.imageCanvas.height
              //convert radastatus to world corrdingates
              let radarWorldY = 1
            
              let radar_dot = [-radarStatus[0], radarWorldY, radarStatus[1], 1]
              let radar_dot_top = [-radarStatus[0], radarWorldY-2, radarStatus[1], 1]
              
              //Adjust this matrix if radar position is greatly different to camera
              let transformation_matrix = [[ 1,  0,  0,  0],
                                            [ 0,  1,  0,  0],
                                            [ 0,  0,  1,  0],
                                            [ 0,  0,  0,  1]];   
                                            
              //CAMERA INTRINSICS ----------------------------------------------
              let K_rgb = [[1060.7331913771368, 0, 936.1470691648806],
                            [0, 1061.7072593533435, 569.0462088683403],
                            [0, 0, 1]];
              //----------------------------------------------------------------

              let transformed_radar_dot = matrixMultiplication(transformation_matrix,transpose([radar_dot]));
              let transformed_radar_dot_top = matrixMultiplication(transformation_matrix,transpose([radar_dot_top]));
              
              transformed_radar_dot.pop()
              transformed_radar_dot_top.pop()

              let unnormalized_image_cord = matrixMultiplication( K_rgb,transformed_radar_dot);
              let unnormalized_image_cord_top = matrixMultiplication( K_rgb,transformed_radar_dot_top);
        
              let x = unnormalized_image_cord[0][0]/unnormalized_image_cord[2][0]
              let y = unnormalized_image_cord[1][0]/unnormalized_image_cord[2][0]
              //let x_top = unnormalized_image_cord_top[0][0]/unnormalized_image_cord_top[2][0]
              let y_top = unnormalized_image_cord_top[1][0]/unnormalized_image_cord_top[2][0]

              let bar_height = y_top - y
              
              //draw the bar
              this.imageContext.fillStyle =  `hsl(
                ${radarStatus[2]},
                50%,
                50%)`;
              this.imageContext.fillRect((x/image.height)*radarHeight, (y/image.width)*radarWidth, bar_height/10, bar_height)
              
            }
          }    
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
