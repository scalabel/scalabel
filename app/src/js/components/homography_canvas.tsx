import { withStyles } from '@material-ui/styles'
import * as React from 'react'
import * as THREE from 'three'
import Session from '../common/session'
import { LabelTypeName } from '../common/types'
import { Grid3D } from '../drawable/3d/grid3d'
import { Plane3D } from '../drawable/3d/plane3d'
import { isCurrentFrameLoaded, isFrameLoaded } from '../functional/state_util'
import { HomographyViewerConfigType, State } from '../functional/types'
import { imageViewStyle } from '../styles/label'
import { clearCanvas, drawImageOnCanvas } from '../view_config/image'
import { ImageCanvas, Props } from './image_canvas'

/** Get basis matrix for use with homography */
function getBasisMatrix (homogeneousPoints: THREE.Vector3[]) {
  const homogeneousMatrix = new THREE.Matrix3()

  for (let i = 0; i < 3; i++) {
    const offset = i * 3
    homogeneousMatrix.elements[offset] = homogeneousPoints[i].x
    homogeneousMatrix.elements[offset + 1] = homogeneousPoints[i].y
    homogeneousMatrix.elements[offset + 2] = 1
  }

  const homogeneousInverse = new THREE.Matrix3()
  homogeneousInverse.getInverse(homogeneousMatrix)

  const target = new THREE.Vector3()
  target.copy(homogeneousPoints[3])
  target.applyMatrix3(homogeneousInverse)
  const scalars = target.toArray()

  const basis = new THREE.Matrix3()
  basis.copy(homogeneousMatrix)

  for (let i = 0; i < 9; i += 3) {
    for (let j = 0; j < 3; j++) {
      basis.elements[i + j] *= scalars[i / 3]
    }
  }

  return basis
}

/**
 * Component for displaying birds eye view homography
 */
class HomographyCanvas extends ImageCanvas {
  /** image */
  private _image?: HTMLImageElement
  /** selected plane */
  private _plane: Plane3D | null
  /** intrinsic matrix */
  private _intrinsicProjection: THREE.Matrix3
  /** inverse of intrinsic */
  private _intrinsicInverse: THREE.Matrix3
  /** homography matrix */
  private _homographyMatrix: THREE.Matrix3
  /** canvas for drawing image & getting colors */
  private _hiddenCanvas: HTMLCanvasElement
  /** context of image canvas */
  private _hiddenContext: CanvasRenderingContext2D | null
  /** image data */
  private _imageData: Uint8ClampedArray | null

  constructor (props: Props) {
    super(props)
    this._plane = null
    this._intrinsicProjection = new THREE.Matrix3()
    this._intrinsicInverse = new THREE.Matrix3()
    this._homographyMatrix = new THREE.Matrix3()
    this._hiddenCanvas = document.createElement('canvas')
    this._hiddenContext = null
    this._imageData = null
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
          const item = this.state.user.select.item
          const sensor = this.state.user.viewerConfigs[this.props.id].sensor
          if (displayRect.width
            && displayRect.height
            && isFrameLoaded(this.state, item, sensor)
            && this.imageContext) {
              // set canvas size
            canvas.style.height = displayRect.height + 'px'
            canvas.style.width = displayRect.width + 'px'
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
    if (this.imageCanvas && this.imageContext) {
      const item = this.state.user.select.item
      const sensor = this.state.user.viewerConfigs[this.props.id].sensor
      if (isFrameLoaded(this.state, item, sensor) &&
          item < Session.images.length &&
          sensor in Session.images[item]) {
        this._image = Session.images[item][sensor]
        // redraw imageCanvas
        if (this._plane) {
          this.drawHomography()
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
   * Override update state function
   * @param state
   */
  protected updateState (state: State): void {
    super.updateState(state)

    const selectedLabel = Session.label3dList.selectedLabel
    if (selectedLabel && selectedLabel.label.type === LabelTypeName.PLANE_3D) {
      this._plane = selectedLabel as Plane3D
    }

    if (this._plane && this.props.id in this.state.user.viewerConfigs) {
      const viewerConfig = this.state.user.viewerConfigs[this.props.id] as
        HomographyViewerConfigType
      const sensorId = viewerConfig.sensor
      const item = state.user.select.item
      if (isFrameLoaded(state, item, sensorId)) {
        if (this._image !== Session.images[item][sensorId]) {
          this._image = Session.images[item][sensorId]
          if (!this._hiddenContext) {
            this._hiddenCanvas.width = this._image.width
            this._hiddenCanvas.height = this._image.height
            this._hiddenContext = this._hiddenCanvas.getContext('2d')
          }
          if (this._hiddenContext) {
            this._hiddenContext.drawImage(this._image, 0, 0)
            this._imageData = this._hiddenContext.getImageData(
              0, 0, this._hiddenCanvas.width, this._hiddenCanvas.height
            ).data
          }
        }
      }

      if (this._image && sensorId in this.state.task.sensors) {
        const sensor = this.state.task.sensors[sensorId]
        if (sensor.intrinsics &&
            sensor.extrinsics &&
            isCurrentFrameLoaded(state, sensorId)) {
          const image =
            Session.images[item][sensorId]

          // Set intrinsics
          const intrinsics = sensor.intrinsics
          const fx = intrinsics.focalLength.x / image.width
          const cx = intrinsics.focalCenter.x / image.width
          const fy = intrinsics.focalLength.y / image.height
          const cy = intrinsics.focalCenter.y / image.height
          this._intrinsicProjection.set(
            fx, 0, cx,
            0, fy, cy,
            0, 0, 1
          )
          this._intrinsicInverse.getInverse(this._intrinsicProjection)

          // Extrinsics
          const extrinsicTranslation = new THREE.Vector3(
            sensor.extrinsics.translation.x,
            sensor.extrinsics.translation.y,
            sensor.extrinsics.translation.z
          )
          const extrinsicQuaternion = new THREE.Quaternion(
            sensor.extrinsics.rotation.x,
            sensor.extrinsics.rotation.y,
            sensor.extrinsics.rotation.z,
            sensor.extrinsics.rotation.w
          )
          const extrinsicQuaternionInverse = extrinsicQuaternion.inverse()

          const grid = this._plane.shapes()[0] as Grid3D

          const sourcePoints = []
          for (let y = 0.5; y >= -0.5; y--) {
            for (let x = 0.5; x >= -0.5; x--) {
              const point = new THREE.Vector3(y, x, 0)
              point.applyMatrix4(grid.matrixWorld)
              point.sub(extrinsicTranslation)
              point.applyQuaternion(extrinsicQuaternionInverse)
              point.applyMatrix3(this._intrinsicProjection)
              point.multiplyScalar(1.0 / point.z)
              sourcePoints.push(point)
            }
          }
          const sourceBasis = getBasisMatrix(sourcePoints)
          const sourceBasisInverse = new THREE.Matrix3()
          sourceBasisInverse.getInverse(sourceBasis)

          const destinationPoints = [
            new THREE.Vector3(0, 0, 1),
            new THREE.Vector3(1, 0, 1),
            new THREE.Vector3(0, 1, 1),
            new THREE.Vector3(1, 1, 1)
          ]
          const destinationBasis = getBasisMatrix(destinationPoints)

          this._homographyMatrix.multiplyMatrices(
            destinationBasis, sourceBasisInverse
          )
        }
      }
    }
  }

  /**
   * Draw image with birds eye view homography
   */
  private drawHomography () {
    if (this.imageCanvas && this.imageContext && this._imageData) {
      if (this._plane && this._image) {
        const homographyData = this.imageContext.createImageData(
          this.imageCanvas.width, this.imageCanvas.height
        )
        const homographyInverse = new THREE.Matrix3()
        homographyInverse.getInverse(this._homographyMatrix)
        for (let dstX = 0; dstX < this.imageCanvas.width; dstX++) {
          for (let dstY = 0; dstY < this.imageCanvas.height; dstY++) {
            // Get source coordinates
            const src = new THREE.Vector3(
              dstX / this.imageCanvas.width, dstY / this.imageCanvas.height, 1
            )
            // src.applyMatrix3(this._intrinsicInverse)
            src.applyMatrix3(homographyInverse)
            // src.applyMatrix3(this._intrinsicProjection)
            src.multiplyScalar(1. / src.z)

            const srcX = Math.floor(
              src.x * this._hiddenCanvas.width
            )
            const srcY = Math.floor(
              src.y * this._hiddenCanvas.height
            )

            if (
                srcX >= 0 &&
                srcY >= 0 &&
                srcX < this._hiddenCanvas.width &&
                srcY < this._hiddenCanvas.height
              ) {
              const imageStart = (srcY * this._hiddenCanvas.width + srcX) * 4
              const homographyStart = (dstY * this.imageCanvas.width + dstX) * 4
              for (let i = 0; i < 4; i++) {
                homographyData.data[homographyStart + i] =
                  this._imageData[imageStart + i]
              }
            }
          }
        }

        this.imageContext.putImageData(homographyData, 0, 0)
      } else {
        clearCanvas(this.imageCanvas, this.imageContext)
      }
    }
  }
}

export default withStyles(
  imageViewStyle, { withTheme: true }
)(HomographyCanvas)
