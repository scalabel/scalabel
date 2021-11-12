import { createStyles, StyleRules } from "@mui/styles"
import { Theme } from "@mui/material/styles"

const fullWidth = 700

type createStyleKey = "listRoot" | "listHeader"
type projectListStyleKey = "coloredListItem"
type formStyleKey =
  | "hidden"
  | "root"
  | "fullWidthText"
  | "halfWidthText"
  | "formGroup"
  | "selectEmpty"
  | "submitButton"
type checkboxStyleKey = "root" | "checked"

// Styles for the create page
export const createStyle = (): StyleRules<{}, createStyleKey> =>
  createStyles({
    listRoot: {
      width: "90%",
      marginLeft: "5%"
    },
    listHeader: {
      textAlign: "center",
      fontWeight: "bold"
    }
  })

// Styles for sidebar project list
export const projectListStyle = (
  theme: Theme
): StyleRules<{}, projectListStyleKey> =>
  createStyles({
    coloredListItem: {
      backgroundColor: theme.palette.action.hover
    }
  })

// Styles for the create form
export const formStyle = (theme: Theme): StyleRules<{}, formStyleKey> =>
  createStyles({
    root: {
      paddingLeft: theme.spacing(3),
      paddingRight: theme.spacing(3)
    },
    fullWidthText: {
      width: fullWidth
    },

    halfWidthText: {
      width: fullWidth / 2
    },

    formGroup: {
      marginTop: theme.spacing(1)
    },

    selectEmpty: {
      width: (fullWidth - Number.parseInt(theme.spacing(1))) / 2,
      marginRight: theme.spacing(1)
    },

    submitButton: {
      marginRight: theme.spacing(1)
    },

    hidden: {
      visibility: "hidden"
    }
  })

// Styles for the upload buttons
export const uploadStyle = createStyles({
  root: {
    width: fullWidth / 5
  },

  button: {
    // padding: 5,
    // marginRight: 10,
    textTransform: "initial"
  },

  textField: {
    width: 130
  },

  filenameText: {
    fontSize: 14,
    overflow: "hidden",
    textOverflow: "ellipsis"
  },

  grid: {
    marginTop: 5
  }
})

// Attribute upload override styling
export const attributeStyle = createStyles({
  root: {
    position: "absolute",
    marginLeft: (fullWidth * 4) / 5
  }
})

export const checkboxStyle = (theme: Theme): StyleRules<{}, checkboxStyleKey> =>
  createStyles({
    root: {
      "&$checked": {
        color: theme.palette.secondary.dark
      }
    },

    checked: {}
  })
