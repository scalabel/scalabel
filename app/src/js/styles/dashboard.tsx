import { Theme } from '@material-ui/core'
import { cyan } from '@material-ui/core/colors'
import createStyles from '@material-ui/core/styles/createStyles'
import { CSSProperties } from '@material-ui/core/styles/withStyles'
/* Dashboard window styles */
export const dashboardWindowStyles = (theme: Theme) =>
  createStyles({
    root: {
      paddingLeft: theme.spacing(2),
      paddingRight: theme.spacing(2),
      paddingTop: theme.spacing(1)
    },
    row: {
      background: theme.palette.common.white
    },
    linkButton: {
      color: cyan[500]
    },
    headerCell: {
      fontWeight: 'bold',
      fontSize: '0.8rem',
      color: theme.palette.common.black
    },
    bodyCell : {
      paddingTop: 0,
      paddingBottom: 0
    }
  })
export const headerStyle = (theme: Theme) =>
  createStyles({
    grow: {
      flexGrow: 1
    },
    chip: {
      marginRight: theme.spacing(2),
      marginLeft: theme.spacing(1)
    }
  })
export const sidebarStyle = (theme: Theme) =>
  createStyles({
    listRoot: {
      marginTop: theme.spacing(2),
      width: '90%',
      marginLeft: '5%'
    },
    listItem: {
      textAlign: 'center',
      margin: 0,
      paddingTop: 2,
      paddingBottom: 2
    },
    coloredListItem: {
      backgroundColor: theme.palette.secondary.light
    },
    link: {
      textAlign: 'center',
      marginTop: theme.spacing(2)
    }
  })
export const listEntryStyle = () =>
  createStyles({
    listTag: {
      textAlign: 'right',
      fontWeight: 'bold'
    },
    listEntry: {
      textAlign: 'left'
    },
    listContainer: {
      margin: 0
    }
  })
/* styles for worker and admin dashboard*/
export const dashboardStyles = (theme: Theme) => createStyles({
  adminRoot: {
    paddingLeft: theme.spacing(3),
    paddingRight: theme.spacing(3)
  },
  workerRoot: {
    flexGrow: 1,
    paddingLeft: theme.spacing(3),
    paddingRight: theme.spacing(3)
  },
  labelText: {
    marginTop: theme.spacing(2)
  },
  appBarSpacer: theme.mixins.toolbar as CSSProperties
})
/* dashboard header style */
export const dashboardHeaderStyles = createStyles({
  title: {
    flexGrow: 1
  }
})
/* tableStyles */
export const tableStyles = (theme: Theme) => createStyles({
  root: {
  },
  headerCell: {
    fontWeight: 'bold',
    fontSize: '0.8rem',
    color: theme.palette.common.black
  },
  row: {
    background: theme.palette.common.white
  }
})
/* tableCellStyles */
export const tableCellStyles = (theme: Theme) => createStyles({
  head: {
    backgroundColor: theme.palette.primary.dark,
    color: theme.palette.common.white
  },
  body: {
    fontSize: 16
  }
})
