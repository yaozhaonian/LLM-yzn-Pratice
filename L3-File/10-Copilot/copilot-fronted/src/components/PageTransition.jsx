// 页面过渡
/**
 * 页面切换动画容器组件，基于 framer-motion 实现路由页面入场、离场平滑过渡：
 * 进入页面：从右侧滑入 + 淡入
 * 离开页面：向左侧滑出 + 淡出
 * 所有页面内容丢进 <PageTransition> 包裹，自动拥有统一转场动画。
 */
import { motion } from 'framer-motion';

const pageVariants = {
  initial: { opacity: 0, x: 50 },
  in: { opacity: 1, x: 0 },
  out: { opacity: 0, x: -50 }
};

export default function PageTransition({ children }) {
  return (
    <motion.div
      initial="initial"
      animate="in"
      exit="out"
      variants={pageVariants}
      transition={{ duration: 0.3 }}
    >
      {children}
    </motion.div>
  );
}






