import CssBaseline from "@material-ui/core/CssBaseline"
import { withStyles } from "@material-ui/core/styles"
import { Stack } from "@mui/material"
import * as React from "react"
import SplitPane from "../common/split_pane/SplitPane"
import Session from "../common/session"
import { LayoutStyles } from "../styles/label"
import LabelPane from "./label_pane"
import PlayerControl from "./player_control"
import { CustomAlert } from "../components/alert"
import { Severity } from "../types/common"

/**
 * Check whether a react node is empty
 *
 * @param node
 */
function isNodeEmpty(node: React.ReactNode): boolean {
  return (
    node === null ||
    node === undefined ||
    node === "" ||
    node === false ||
    node === {}
  )
}

interface ClassType {
  /** title bar */
  titleBar: string
  /** everything below title bar */
  main: string
  /** interface container */
  interfaceContainer: string
  /** pane container */
  paneContainer: string
  /** alerts */
  alerts: string
}

interface Props {
  /** The title bar */
  titleBar: React.ReactNode
  /** The top part of the left side bar */
  leftSidebar1: React.ReactNode
  /** The bottom part of the left side bar */
  leftSidebar2: React.ReactNode
  /** The bottom bar */
  bottomBar: React.ReactNode
  /** The top part of the right side bar */
  rightSidebar1: React.ReactNode
  /** The bottom part of the right side bar */
  rightSidebar2: React.ReactNode
  /** class type */
  classes: ClassType
}

interface LayoutState {
  /** The width of the left side bar */
  left_size: number
  /** The height of the center side bar */
  center_size: number
  /** The width of the right side bar */
  right_size: number
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
;(window as any).__MUI_USE_NEXT_TYPOGRAPHY_VARIANTS__ = true

/**
 * Layout of the labeling interface
 */
class LabelLayout extends React.Component<Props, LayoutState> {
  /**
   * Constructor
   *
   * @param props
   */
  constructor(props: Props) {
    super(props)
    this.state = { left_size: 0, center_size: 0, right_size: 0 }
    Session.subscribe(this.onStateUpdated.bind(this))
  }

  /**
   * called on redux store update
   */
  public onStateUpdated(): void {
    this.setState((state: LayoutState): LayoutState => {
      return state
    })
  }

  /**
   * Handler on change
   *
   * @param {number} size
   * @param {string} position
   */
  public handleOnChange(size: number, position: string): void {
    this.setState((state: LayoutState): LayoutState => {
      const newState = { ...state }
      if (position === "left" && state.left_size !== size) {
        newState.left_size = size
      } else if (position === "center" && state.center_size !== size) {
        newState.center_size = size
      } else if (position === "right" && state.right_size !== size) {
        newState.right_size = size
      }
      return newState
    })
  }

  /**
   * Split component with the second component optional
   *
   * @param {string} split - horizontal or vertical
   * @param {React.Fragment} comp1 - the first component
   * @param {React.Fragment} comp2 - the second component
   * @param {string} name1 - the class name of the first component
   * @param {string} name2 - the class name of the second component
   * @param {number} min - the minimum size
   * @param {number} dflt - the default size
   * @param {number} max - the maximum size
   * @param {string} primary - which component the size constraint is for
   * the second component
   * @param {string} position - left, center or right:
   * which size to update in layoutState
   * @return {Component}
   */
  public split(
    split: "vertical" | "horizontal",
    comp1: React.ReactNode,
    comp2: React.ReactNode,
    name1: string,
    name2: string,
    min: number,
    dflt: number,
    max: number,
    primary: "first" | "second" = "first",
    position: string = "center"
  ): React.ReactNode {
    if (isNodeEmpty(comp1)) {
      return
    }
    return !isNodeEmpty(comp2) ? (
      <SplitPane
        split={split}
        minSize={min}
        defaultSize={dflt}
        maxSize={max}
        primary={primary}
        pane1Style={{ overflowY: "auto", overflowX: "auto" }}
        onChange={(size) => {
          this.handleOnChange(size, position)
        }}
      >
        <div className={name1}>{comp1}</div>
        <div className={name2}>{comp2}</div>
      </SplitPane>
    ) : (
      <div className={name1}>{comp1}</div>
    )
  }

  /** Return alerts */
  protected getAlerts(): [] | JSX.Element[] {
    const alerts: React.ReactElement[] = []
    const state = Session.getState()
    if (state.session.alerts !== []) {
      state.session.alerts.forEach((alert) => {
        alerts.push(
          <CustomAlert
            key={`alert${alert.id}`}
            id={alert.id}
            severity={alert.severity as Severity}
            msg={alert.message}
            timeout={alert.timeout}
          />
        )
      })

      return alerts
    }

    return []
  }

  /**
   * Render function
   *
   * @return {React.Fragment} React fragment
   */
  public render(): React.ReactFragment {
    const {
      titleBar,
      leftSidebar1,
      leftSidebar2,
      bottomBar,
      rightSidebar1,
      rightSidebar2,
      classes
    } = this.props
    const leftDefaultWidth = 160
    const leftMaxWidth = 180
    const leftMinWidth = 140
    const rightDefaultWidth = 200
    const rightMaxWidth = 300
    const rightMinWidth = 180
    const topDefaultHeight = 200
    const topMaxHeight = 300
    const topMinHeight = 180
    const bottomDefaultHeight = 200
    const bottomMaxHeight = 300
    const bottomMinHeight = 180

    const playerControl = (
      <PlayerControl
        key="player-control"
        numFrames={Session.getState().task.items.length}
        keyInterval={Session.getState().task.config.keyInterval}
      />
    )

    const state = Session.getState()

    const labelInterface = (
      <div className={this.props.classes.interfaceContainer}>
        <div className={this.props.classes.paneContainer}>
          <LabelPane pane={state.user.layout.rootPane} key={"rootPane"} />
        </div>
        {playerControl}
      </div>
    )

    return (
      <React.Fragment>
        <CssBaseline />
        <div className={classes.titleBar}>{titleBar}</div>
        <div className={classes.alerts}>
          <Stack spacing={1}>{this.getAlerts()}</Stack>
        </div>
        <main className={classes.main}>
          {this.split(
            "vertical",
            // Left sidebar
            this.split(
              "horizontal",
              leftSidebar1,
              leftSidebar2,
              "leftSidebar1",
              "leftSidebar2",
              topMinHeight,
              topDefaultHeight,
              topMaxHeight,
              "first"
            ),

            this.split(
              "vertical",
              // Center
              this.split(
                "horizontal",
                labelInterface,
                bottomBar,
                "main",
                "bottomBar",
                bottomMinHeight,
                bottomDefaultHeight,
                bottomMaxHeight,
                "second",
                "center"
              ),

              // Right sidebar
              this.split(
                "horizontal",
                rightSidebar1,
                rightSidebar2,
                "rightSidebar1",
                "rightSidebar2",
                topMinHeight,
                topDefaultHeight,
                topMaxHeight
              ),

              "center",
              "rightSidebar",
              rightMinWidth,
              rightDefaultWidth,
              rightMaxWidth,
              "second",
              "right"
            ),

            "leftSidebar",
            "centerAndRightSidebar",
            leftMinWidth,
            leftDefaultWidth,
            leftMaxWidth,
            "first",
            "left"
          )}
        </main>
        {/* End footer */}
      </React.Fragment>
    )
  }
}

export default withStyles(LayoutStyles, { withTheme: true })(LabelLayout)
