import { withStyles } from "@material-ui/styles"
import * as React from "react"
import { connect } from "react-redux"
import * as THREE from "three"

import Session from "../common/session"
import { LabelTypeName } from "../const/common"
import { Plane3D } from "../drawable/3d/plane3d"
import { isCurrentFrameLoaded, isFrameLoaded } from "../functional/state_util"
import { imageViewStyle } from "../styles/label"
import { HomographyViewerConfigType, State } from "../types/state"
import { clearCanvas, drawImageOnCanvas } from "../view_config/image"
import { ImageCanvas, Props } from "./image_canvas"
import { mapStateToDrawableProps } from "./viewer"

/**
 * Component for displaying birds eye view homography
 */
class HomographyCanvas extends ImageCanvas {
  /** image */
  private _image?: HTMLImageElement
  /** selected plane */
  private _plane: Plane3D | null
  /** previous plane state */
  private _planeState: string
  /** updated */
  private _updated: boolean
  /** intrinsic matrix */
  private readonly _intrinsicMatrix: THREE.Matrix4
  /** homography matrix */
  private readonly _homographyMatrix: THREE.Matrix3
  /** canvas for drawing image & getting colors */
  private _hiddenCanvas: HTMLCanvasElement
  /** context of image canvas */
  private _hiddenContext: CanvasRenderingContext2D | null
  /** image data */
  private _imageData: Uint8ClampedArray | null
  /** image canvas width */
  private _canvasWidth: number

  /**
   * Constructor
   *
   * @param props
   */
  constructor(props: Props) {
    super(props)
    this._plane = null
    this._intrinsicMatrix = new THREE.Matrix4()
    this._homographyMatrix = new THREE.Matrix3()
    this._hiddenCanvas = document.createElement("canvas")
    this._hiddenContext = null
    this._imageData = null
    this._planeState = ""
    this._updated = false
    this._canvasWidth = 0
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
            if (canvas.width !== this._canvasWidth) {
              this._canvasWidth = canvas.width
              this._updated = true
            }
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
              // Set canvas size
              canvas.style.height = `${displayRect.height}px`
              canvas.style.width = `${displayRect.width}px`
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
      const sensor = this.state.user.viewerConfigs[this.props.id].sensor
      if (
        isFrameLoaded(this.state, item, sensor) &&
        item < Session.images.length &&
        sensor in Session.images[item]
      ) {
        this._image = Session.images[item][sensor]
        // Redraw imageCanvas
        if (this._plane !== null) {
          if (this._updated) {
            this.drawHomography()
            this._updated = false
          }
        } else {
          drawImageOnCanvas(this.imageCanvas, this.imageContext, this._image)
        }
      } else {
        clearCanvas(this.imageCanvas, this.imageContext)
      }
    }
    return true
  }

  /**
   * Update homography matrix
   *
   * @param state
   */
  protected updateHomography(state: State): void {
    const sensorId = Object.keys(state.task.sensors).map((key) =>
      parseInt(key)
    )[0]

    const sensor = this.state.task.sensors[sensorId]
    const intrinsics = sensor?.intrinsics ?? null
    if (this._plane !== null && intrinsics !== null) {
      // Set intrinsics
      const fx = intrinsics.focalLength.x
      const cx = intrinsics.focalCenter.x
      const fy = intrinsics.focalLength.y
      const cy = intrinsics.focalCenter.y
      const intrinsicArr = [fx, 0, 0, 0, 0, fy, 0, 0, cx, cy, 1, 0, 0, 0, 0, 0]
      this._intrinsicMatrix.fromArray(intrinsicArr)

      const grid = this._plane.internalShapes()[0]
      const matrix = new THREE.Matrix4()
      matrix.makeRotationFromQuaternion(grid.quaternion)
      matrix.setPosition(grid.position.x, grid.position.y, grid.position.z)

      const extrinsics = new THREE.Matrix4()
      extrinsics.copy(matrix)

      const projection = new THREE.Matrix4()
      projection.multiplyMatrices(this._intrinsicMatrix, extrinsics)
      const projArray = projection.toArray()
      const indices = [0, 1, 2, 4, 5, 6, 12, 13, 14]
      const homographyValues = indices.map((i) => projArray[i])
      this._homographyMatrix.fromArray(homographyValues)
    }
  }

  /**
   * Override update state function
   *
   * @param state
   */
  protected updateState(state: State): void {
    super.updateState(state)

    const labels = Session.label3dList.labels()
    const plane =
      labels.filter(
        (l) =>
          l.item === state.user.select.item &&
          l.label.type === LabelTypeName.PLANE_3D
      )[0] ?? null
    if (plane !== null) {
      this._plane = plane as Plane3D
      const planeState = JSON.stringify(plane.internalShapes()[0].toState())
      if (this._planeState !== planeState) {
        this._updated = true
        this._planeState = planeState
      }
    } else {
      this._plane = null
      this._planeState = ""
      this._updated = true
    }

    if (
      this._plane !== null &&
      this.props.id in this.state.user.viewerConfigs &&
      this._updated
    ) {
      const viewerConfig = this.state.user.viewerConfigs[
        this.props.id
      ] as HomographyViewerConfigType
      const sensorId = viewerConfig.sensor
      const item = state.user.select.item
      if (isFrameLoaded(state, item, sensorId)) {
        if (this._image !== Session.images[item][sensorId]) {
          this._image = Session.images[item][sensorId]
          if (this._hiddenContext === null) {
            this._hiddenCanvas.width = this._image.width
            this._hiddenCanvas.height = this._image.height
            this._hiddenContext = this._hiddenCanvas.getContext("2d")
          }
          if (this._hiddenContext !== null) {
            this._hiddenContext.drawImage(this._image, 0, 0)
            this._imageData = this._hiddenContext.getImageData(
              0,
              0,
              this._hiddenCanvas.width,
              this._hiddenCanvas.height
            ).data
          }
        }
      }

      if (isCurrentFrameLoaded(state, sensorId)) {
        this.updateHomography(state)
      }
    }
  }

  /**
   * Draw image with birds eye view homography
   */
  private drawHomography(): void {
    if (
      this.imageCanvas !== null &&
      this.imageContext !== null &&
      this._imageData !== null &&
      this._plane !== null &&
      this._image !== null
    ) {
      const homographyData = this.imageContext.createImageData(
        this.imageCanvas.width,
        this.imageCanvas.height
      )

      const width = 30
      const height = 30
      for (let dstX = 0; dstX < this.imageCanvas.width; dstX++) {
        for (let dstY = 0; dstY < this.imageCanvas.height; dstY++) {
          // Get source coordinates
          const src = new THREE.Vector3(
            (dstX * width) / this.imageCanvas.width - width / 2,
            (dstY * height) / this.imageCanvas.height - height / 2,
            1
          )
          src.applyMatrix3(this._homographyMatrix)
          const z = src.z
          src.multiplyScalar(1 / z)

          const srcX = Math.floor(src.x)
          const srcY = Math.floor(src.y)
          if (
            z > 0 &&
            srcX >= 0 &&
            srcY >= 0 &&
            srcX < this._hiddenCanvas.width &&
            srcY < this._hiddenCanvas.height
          ) {
            const imageStart = (srcY * this._hiddenCanvas.width + srcX) * 4
            const homographyStart =
              ((this.imageCanvas.height - dstY - 1) * this.imageCanvas.width +
                dstX) *
              4
            for (let i = 0; i < 4; i++) {
              homographyData.data[homographyStart + i] =
                this._imageData[imageStart + i]
            }
          }
        }
      }

      this.imageContext.putImageData(homographyData, 0, 0)
    }
  }
}

const styledCanvas = withStyles(imageViewStyle, { withTheme: true })(
  HomographyCanvas
)
export default connect(mapStateToDrawableProps)(styledCanvas)
