import createStyles from '@material-ui/core/styles/createStyles'
import { withStyles } from '@material-ui/core/styles/index'
import * as React from 'react'
import * as THREE from 'three'
import Session from '../common/session'
import { isCurrentItemLoaded } from '../functional/state_util'
import { PointCloudViewerConfigType, State } from '../functional/types'
import { DrawableCanvas } from './viewer'

const styles = () => createStyles({
  point_cloud_canvas: {
    position: 'absolute',
    height: '100%',
    width: '100%'
  }
})

interface ClassType {
  /** CSS canvas name */
  point_cloud_canvas: string
}

interface Props {
  /** CSS class */
  classes: ClassType
  /** container */
  display: HTMLDivElement | null
  /** viewer id */
  id: number
  /** camera */
  camera: THREE.PerspectiveCamera
}

/**
 * Canvas Viewer
 */
class PointCloudCanvas extends DrawableCanvas<Props> {
  /** Container */
  private display: HTMLDivElement | null
  /** Canvas to draw on */
  private canvas: HTMLCanvasElement | null
  /** ThreeJS Renderer */
  private renderer?: THREE.WebGLRenderer
  /** ThreeJS Scene object */
  private scene: THREE.Scene
  /** ThreeJS Camera */
  private camera: THREE.PerspectiveCamera
  /** ThreeJS sphere mesh for indicating camera target location */
  private target: THREE.AxesHelper
  /** Current point cloud for rendering */
  private pointCloud: THREE.Points | null
  /** drawable callback */
  private _drawableUpdateCallback: () => void

  /**
   * Constructor, ons subscription to store
   * @param {Object} props: react props
   */
  constructor (props: Readonly<Props>) {
    super(props)
    this.scene = new THREE.Scene()
    this.camera = props.camera
    this.target = new THREE.AxesHelper(0.5)
    this.scene.add(this.target)

    this.pointCloud = null

    this.canvas = null
    this.display = null

    this._drawableUpdateCallback = this.renderThree.bind(this)
  }

  /** mount callback */
  public componentDidMount () {
    super.componentDidMount()
    Session.label3dList.subscribe(this._drawableUpdateCallback)
  }

  /** mount callback */
  public componentWillUnmount () {
    super.componentWillUnmount()
    Session.label3dList.unsubscribe(this._drawableUpdateCallback)
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render () {
    const { classes } = this.props

    let canvas = (
      <canvas
        key={`point-cloud-canvas-${this.props.id}`}
        className={classes.point_cloud_canvas}
        ref={(ref) => { this.initializeRefs(ref) }}
      />
    )

    if (this.display) {
      const displayRect = this.display.getBoundingClientRect()
      canvas = React.cloneElement(
        canvas,
        { height: displayRect.height, width: displayRect.width }
      )
    }

    this.redraw()

    return canvas
  }

  /**
   * Handles canvas redraw
   * @return {boolean}
   */
  public redraw (): boolean {
    const state = this.state
    if (isCurrentItemLoaded(state) && this.canvas) {
      const item = state.user.select.item
      const sensor =
        this.state.user.viewerConfigs[this.props.id].sensor
      if (item < Session.images.length && sensor in Session.pointClouds[item]) {
        this.updateRenderer()
        this.pointCloud = Session.pointClouds[item][sensor]
        this.renderThree()
      }
    }
    return true
  }

  /**
   * Override method
   * @param _state
   */
  protected updateState (state: State) {
    if (this.display !== this.props.display) {
      this.display = this.props.display
      this.forceUpdate()
    }
    const config =
      state.user.viewerConfigs[this.props.id] as PointCloudViewerConfigType
    this.target.position.set(config.target.x, config.target.y, config.target.z)
  }

  /**
   * Render ThreeJS Scene
   */
  private renderThree () {
    if (this.renderer && this.pointCloud) {
      this.scene.children = []
      this.scene.add(this.pointCloud)
      this.scene.add(this.target)
      this.renderer.render(this.scene, this.camera)
    }
  }
  /**
   * Set references to div elements and try to initialize renderer
   * @param {HTMLDivElement} component
   * @param {string} componentType
   */
  private initializeRefs (component: HTMLCanvasElement | null) {
    if (!component) {
      return
    }

    if (component.nodeName === 'CANVAS') {
      if (this.canvas !== component) {
        this.canvas = component
        const rendererParams = { canvas: this.canvas, alpha: true }
        this.renderer = new THREE.WebGLRenderer(rendererParams)
        this.forceUpdate()
      }

      if (isCurrentItemLoaded(this.state)) {
        this.updateRenderer()
      }
    }
  }

  /**
   * Update rendering constants
   */
  private updateRenderer () {
    if (this.canvas && this.renderer) {
      this.renderer.setSize(
        this.canvas.width,
        this.canvas.height
      )
    }
    // if (this.canvas && this.renderer) {
    //   const config = getCurrentViewerConfig(
    //     this.state, this.props.id
    //   ) as PointCloudViewerConfigType
    //   updateThreeCameraAndRenderer(
    //     config,
    //     this.camera,
    //     this.canvas,
    //     this.renderer
    //   )
    // }
  }
}

export default withStyles(styles, { withTheme: true })(PointCloudCanvas)
