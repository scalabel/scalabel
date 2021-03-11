import { blue, grey } from "@material-ui/core/colors"
import { createStyles, StyleRules } from "@material-ui/core/styles"

/**
 * Category style
 */
export const categoryStyle = (): // theme: Theme
StyleRules<"root" | "button" | "formControl" | "primary" | "buttonGroup", {}> =>
  createStyles({
    root: {
      display: "flex",
      flexWrap: "wrap",
      flexDirection: "column",
      alignItems: "left"
    },
    formControl: {
      width: "80%",
      margin: "2px 10%"
    },
    primary: {
      fontWeight: "bold",
      width: "100%",
      textAlign: "center",
      padding: "0px"
    },
    button: {
      height: 22,
      textTransform: "none",
      borderRadius: 0,
      borderWidth: "1px"
    },
    buttonGroup: {
      width: "100%"
    }
  })

/**
 * Switch style
 */
export const switchStyle = (): StyleRules<
  "root" | "primary" | "switchBase" | "checked" | "track",
  {}
> =>
  createStyles({
    root: {
      width: "80%",
      margin: "2px 10%",
      padding: "0px 0px"
    },
    primary: {
      // fontSize: "15px"
      fontWeight: "bold"
    },
    switchBase: {
      color: grey[400],
      "&$checked": {
        color: grey[500],
        "& + $track": {
          backgroundColor: blue[700]
        }
      }
    },
    checked: {},
    track: {}
  })

export const toolbarButtonStyle = (): StyleRules<"root" | "label", {}> =>
  createStyles({
    root: {
      // border: 0,
      // color: "black",
      // height: "80%"
      width: "80%",
      margin: "3px 10%"
      // padding: "5px 15px",
      // boxShadow: "0 1px 0px 5px rgba(250, 250, 250, 1)",
      // fontSize: "15px",
      // background: (props: StyledButtonProps) => props.background,
      // margin: "0px 20px"
    },
    label: {
      // fontSize: "15px"
    }
  })

/**
 * Toggle button style
 */
export const toggleButtonStyle = (): StyleRules<"root" | "label"> =>
  createStyles({
    root: {
      color: "rgba(0, 0, 0, 0.38)",
      height: "28px",
      padding: "1px 2px",
      // fontSize: "15px",
      minWidth: "28px",
      borderRadius: "0px"
    },
    label: {
      // fontSize: "10px"
    }
  })

/**
 * List button style
 */
export const listButtonStyle = (): StyleRules<
  "root" | "toggleContainer" | "buttonGroup" | "primary" | "toggleButton",
  {}
> =>
  createStyles({
    root: {
      padding: "0px",
      height: "28px"
    },
    toggleContainer: {
      display: "flex",
      alignItems: "left",
      justifyContent: "flex-start",
      width: "100%",
      padding: "0px"
      // background: "rgba(250,250,250,0)"
    },
    buttonGroup: {
      width: "100%",
      display: "flex",
      // Needed to prevent compiler from complaining about types
      flexWrap: "wrap" as "wrap"
    },
    primary: {
      // fontSize: "10px",
      padding: "0px",
      textAlign: "left",
      width: "100%",
      fontWeight: "bold"
    },
    toggleButton: {
      flexGrow: 1,
      borderRadius: 0
    }
  })

/**
 * Label2d View style
 */
export const label2dViewStyle = (): StyleRules<
  "label2d_canvas" | "control_canvas" | "hair",
  {}
> =>
  createStyles({
    label2d_canvas: {
      position: "absolute",
      "z-index": 1
    },
    control_canvas: {
      position: "absolute",
      visibility: "hidden",
      "z-index": 2
    },
    hair: {
      position: "fixed",
      "margin-top": "0px",
      "margin-left": "0px",
      background: "transparent",
      "border-top": "1px dotted #0000ff",
      "border-left": "1px dotted #0000ff",
      "pointer-events": "none",
      "z-index": 3
    }
  })

/**
 * Image view style
 */
export const imageViewStyle = (): StyleRules<"image_canvas", {}> =>
  createStyles({
    image_canvas: {
      position: "absolute",
      "z-index": 0
    }
  })

/**
 * Player control style
 */
export const playerControlStyles = (): StyleRules<
  "button" | "underline" | "playerControl" | "input" | "slider",
  {}
> =>
  createStyles({
    button: {
      color: "#bbbbbb",
      left: "-3px",
      verticalAlign: "middle"
    },
    playerControl: {
      display: "block",
      position: "relative",
      top: "calc(100% - 55px)",
      zIndex: 100,
      marginRight: "30px"
    },
    input: {
      background: "#000000",
      color: "green",
      width: "50px",
      fontWeight: 500,
      left: "-1px",
      right: "2px",
      verticalAlign: "middle"
    },
    underline: {
      color: "green"
    },
    slider: {
      selectionColor: "green",
      rippleColor: "white",
      verticalAlign: "middle"
    }
  })

export const LayoutStyles = (): StyleRules<
  "titleBar" | "main" | "interfaceContainer" | "paneContainer",
  {}
> =>
  createStyles({
    titleBar: {
      height: "50px"
    },
    main: {
      height: "calc(100% - 50px)",
      display: "block",
      position: "absolute",
      outline: "none",
      width: "100%"
    },
    interfaceContainer: {
      display: "block",
      height: "100%",
      position: "absolute",
      outline: "none",
      width: "100%",
      background: "#222222"
    },
    paneContainer: {
      width: "100%",
      height: "calc(100% - 60px)",
      position: "absolute",
      top: 0,
      left: 0
    }
  })
