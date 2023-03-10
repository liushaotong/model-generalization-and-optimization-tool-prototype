// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import VueRouter from 'vue-router'
import axios from 'axios'
// import VueAxios from 'vue-axios'

import Complexity from './views/complexity'
import Generalization from './views/generalization'
import Pruning from './views/pruning'

Vue.config.productionTip = false
Vue.use(ElementUI);
Vue.use(VueRouter);
Vue.prototype.$http = axios
// Vue.use(VueAxios, axios)
/* eslint-disable no-new */
const routes = [
  { path: '/complexity', component: Complexity },
  { path: '/generalization', component: Generalization },
  { path: '/pruning', component: Pruning },
];

const router = new VueRouter({
  routes
});



new Vue({
  router,
  el: '#app',
  components: { App },
  template: '<App/>'
})

