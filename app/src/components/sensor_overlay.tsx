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
        for (const sensor of this.state.session.overlayStatus){
          if (isFrameLoaded(this.state, item, sensor) &&
              item < Session.images.length &&
              sensor in Session.images[item]){
              const image = Session.images[item][sensor]
              // Redraw imageCanvas
              if (first){
                drawImageOnCanvas(this.imageCanvas, this.imageContext, image,false)
                first = false
              } else {
                drawImageOnCanvas(this.imageCanvas, this.imageContext, image,true)
              }
          } else if (sensor === 9){
            // project a bar on the image base on the radarstatus
            const image = Session.images[item][0]
            const radarStatus = this.state.session.radarStatus
            if (radarStatus.length > 0){
              const radarWidth = this.imageCanvas.width
              const radarHeight = this.imageCanvas.height
              //convert radastatus to world corrdingates
              let radarWorldZ = 1
              
              //let radar_dot = [radarStatus[1]*radar_scale,radarStatus[0]*radar_scale, radarWorldZ, 1]
              let radar_dot = [radarStatus[0], radarWorldZ, radarStatus[1], 1]
              let radar_dot_top = [radarStatus[0], radarWorldZ-1.2, radarStatus[1], 1]



              //calcualte the position of the bar, hardcoded for now
              //TODO: dont hardcode the conversion matrix
             
              
              // let matrix = [[ 0, -1,  0,  0.106],
              //               [ 0,  0, -1,  0   ],
              //               [ 1,  0,  0, -0.05 ],
              //               [ 0,  0,  0,  1   ]];
              // let matrix = [[ 0, -1,  0,  0.106],
              //               [ 0,  0, -1,  0   ],
              //               [ 1,  0,  0, -1.5 ],
              //               [ 0,  0,  0,  1   ]];
              let matrix = [[ 1,  0,  0,  0],
                            [ 0,  1,  0,  0   ],
                            [ 0,  0,  1,  0 ],
                            [ 0,  0,  0,  1   ]];                           

              //let result = matrixMultiplication([radar_dot], matrix)[0];
              let result = matrixMultiplication(matrix,transpose([radar_dot]));
              let result_top = matrixMultiplication(matrix,transpose([radar_dot_top]));
              let K_rgb = [[1060.7331913771368, 0, 936.1470691648806],
                            [0, 1061.7072593533435, 569.0462088683403],
                            [0, 0, 1]];
              result.pop()
              result_top.pop()
              let result2 = matrixMultiplication( K_rgb,result);
              let result2_top = matrixMultiplication( K_rgb,result_top);
        
              let x = result2[0][0]/result2[2][0]
              let y = result2[1][0]/result2[2][0]
              let x_top = result2_top[0][0]/result2_top[2][0]
              let y_top = result2_top[1][0]/result2_top[2][0]

              console.log(x,y)
              console.log(x_top,y_top)

              let bar_height = y_top - y
              //draw the bar
              this.imageContext.fillStyle =  `hsl(
                ${radarStatus[2]},
                50%,
                50%)`;
              this.imageContext.fillRect((x/image.height)*radarHeight, (y/image.width)*radarWidth, bar_height/10, bar_height)
               
              // const radarBarWidth = radarWidth / radarStatus.length
              // const radarBarHeight = radarHeight / 10
              // for (let i = 0; i < radarStatus.length; i++){
              //   const radarBar = radarStatus[i]
              //   const radarBarX = i * radarBarWidth
              //   const radarBarY = radarHeight - radarBarHeight
              //   this.imageContext.fillStyle = radarBar.color
              //   this.imageContext.fillRect(radarBarX, radarBarY, radarBarWidth, radarBarHeight)
              // } 
            }
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
