import React from 'react';
import PropTypes from 'prop-types';
import CssBaseline from '@material-ui/core/CssBaseline';
import SplitPane from 'react-split-pane';

type Props = {
  titleBar: Object,
  leftSidebar1: Object,
  leftSidebar2?: Object,
  main: Object,
  bottomBar?: Object,
  rightSidebar1?: Object,
  rightSidebar2?: Object,
}

window.__MUI_USE_NEXT_TYPOGRAPHY_VARIANTS__ = true;

/**
 * Split component with the second component optional
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
 * @return {Component}
 */
function optionalSplit(split, comp1, comp2, name1, name2,
                         min, dflt, max, primary) {
  if (!comp1) {
    return;
  }
  return (
      comp2 ?
          <SplitPane split={split} minSize={min}
                     defaultSize={dflt}
                     maxSize={max} primary={primary}>
            <div className={name1}>
              {comp1}
            </div>
            <div className={name2}>
              {comp2}
            </div>
          </SplitPane>
          : <div className={name1}>
            {comp1}
          </div>
  );
}

/**
 * Layout of the labeling interface
 */
class LabelLayout extends React.Component<Props> {
  /**
  * Render function
  * @return {React.Fragment} React fragment
  */
  render() {
    const {titleBar, leftSidebar1, leftSidebar2, bottomBar,
      main, rightSidebar1, rightSidebar2} = this.props;

    const leftDefaultWidth = 200;
    const leftMaxWidth = 300;
    const leftMinWidth = 180;
    const rightDefaultWidth = 200;
    const rightMaxWidth = 300;
    const rightMinWidth = 180;
    const topDefaultHeight = 200;
    const topMaxHeight = 300;
    const topMinHeight = 180;
    const bottomDefaultHeight = 200;
    const bottomMaxHeight = 300;
    const bottomMinHeight = 180;

    return (
        <React.Fragment>
          <CssBaseline />
          <div className='titleBar'>
            {titleBar}
          </div>
          <main>
            <SplitPane split='vertical' minSize={leftMinWidth}
                       defaultSize={leftDefaultWidth}
                       maxSize={leftMaxWidth}>
              {/* left sidebar */}
              {optionalSplit('horizontal', leftSidebar1, leftSidebar2,
                  'leftSidebar1', 'leftSidebar2',
                  topMinHeight, topDefaultHeight, topMaxHeight)}

              {optionalSplit('vertical',
                  // content
                  optionalSplit('horizontal', main, bottomBar,
                    'main', 'bottomBar',
                    bottomMinHeight, bottomDefaultHeight, bottomMaxHeight,
                    'second'),
                  // right sidebar
                  optionalSplit('horizontal', rightSidebar1, rightSidebar2,
                    'rightSidebar1', 'rightSidebar2',
                    topMinHeight, topDefaultHeight, topMaxHeight),
                  'center', 'rightSidebar',
                  rightMinWidth, rightDefaultWidth, rightMaxWidth, 'second'
              )}
            </SplitPane>
          </main>
          {/* End footer */}
        </React.Fragment>
    );
  }
}

LabelLayout.propTypes = {
  titleBar: PropTypes.object.isRequired,
  leftSidebar1: PropTypes.object.isRequired,
  leftSidebar2: PropTypes.object,
  main: PropTypes.object.isRequired,
  bottomBar: PropTypes.object,
  rightSidebar1: PropTypes.object,
  rightSidebar2: PropTypes.object,
};

export default LabelLayout;
