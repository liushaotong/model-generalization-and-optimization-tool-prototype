<template>
  <div class="page-container">
    <el-row>
      <el-col :span="24">
        <el-card class="metrics-card">
          <div slot="header">
            <span class="card-title">泛化性度量结果</span>
          </div>
          <div>
            <el-button type="primary" :loading="loading" @click="loadData">开始度量</el-button>
          </div>
          <div class="metric-item" v-for="(value, key) in metrics" :key="key">
            <div class="metric-name">{{ key }}</div>
            <div class="metric-value">{{ value }}</div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>
<script>
export default {
  data() {
    return {
      metrics: {},
      loading: false
    };
  },
  methods: {
    loadData() {
      this.loading = true;
      this.metrics = {}
      const selectedTask = sessionStorage.getItem('selectedTask');
      let dict = JSON.parse(sessionStorage.getItem('complexityMetrics'));
      dict['selectedTask'] = selectedTask;
      this.$http.post('http://localhost:5000/generalization', dict)
        .then(response => {
          // 保存数据到 session
          sessionStorage.setItem('generalizationMetrics', JSON.stringify(response.data));
          this.metrics = response.data;
          // this.$nextTick(() => {
          //   this.$forceUpdate();
          // });
        })
        .catch(error => {
          console.log(error);
        })
        .finally(() => {
          this.loading = false; // 请求结束后将loading设置为false
        });
    },
    loadFromSession() {
      // 从 session 获取数据
      const metrics = sessionStorage.getItem('generalizationMetrics');
      if (metrics) {
        this.metrics = JSON.parse(metrics);
      }
    }
  },
  created() {
    // 在组件创建时从 session 中加载数据
    this.loadFromSession();
  }
};
</script>
<style scoped>
.page-container {
  padding: 20px;
}

.metrics-card {
  margin-bottom: 20px;
}

.card-title {
  font-size: 18px;
  font-weight: bold;
}

.metric-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #ebeef5;
  padding: 10px 0;
}

.metric-name {
  flex: 1;
  font-size: 16px;
  font-weight: bold;
}

.metric-value {
  font-size: 16px;
  color: #606266;
}
</style>