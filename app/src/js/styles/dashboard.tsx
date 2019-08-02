import { Theme } from '@material-ui/core'
import { cyan } from '@material-ui/core/colors'
import createStyles from '@material-ui/core/styles/createStyles'
import { defaultAppBar } from './general'
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
  root: {
    display: 'flex'
  },
  toolbar: {
    paddingRight: 24
  },
  toolbarIcon: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    padding: '0 8px',
    ...theme.mixins.toolbar
  },
  appBar: {...defaultAppBar,
    zIndex: theme.zIndex.drawer + 1,
    transition: theme.transitions.create(['width', 'margin'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen
    })},
  title: {
    flexGrow: 1
  },
  drawerPaper: {
    position: 'relative',
    whiteSpace: 'nowrap',
    width: 285,
    transition: theme.transitions.create('width', {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.enteringScreen
    })
  },
  appBarSpacer: theme.mixins.toolbar,
  content: {
    flexGrow: 1,
    padding: theme.spacing(3),
    height: '100vh',
    overflow: 'auto'
  },
  chartContainer: {
    marginLeft: -22
  },
  h5: {
    marginBottom: theme.spacing(2)
  }
})

/* tableStyles */
export const tableStyles = (theme: Theme) => createStyles({
  root: {
    width: '100%',
    marginTop: theme.spacing(3),
    overflowX: 'auto'
  },
  table: {
    minWidth: 700
  },
  row: {
    '&:nth-of-type(odd)': {
      backgroundColor: theme.palette.background.default
    }
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
