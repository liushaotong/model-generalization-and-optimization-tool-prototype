<template>
  <div class="page-container">
    <el-row>
      <el-col :span="24">
        <el-card class="metrics-card">
          <div slot="header">
            <span class="card-title">模型复杂度计算结果</span>
          </div>
          <div>
            <el-button type="primary" :loading="loading" @click="loadData">开始计算</el-button>
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
      let dict = {};
      dict['selectedTask'] = selectedTask;
      this.$http.post('http://localhost:5000/complexity', dict)
        .then(response => {
          // 保存数据到 session
          sessionStorage.setItem('complexityMetrics', JSON.stringify(response.data));
          for (let key in response.data) {
            if (response.data.hasOwnProperty(key)) {
              let value = response.data[key];
              this.metrics[key] = Math.abs(value);
            }
          }
          this.$nextTick(() => {
            this.$forceUpdate();
          });
        })
        .catch(error => {
          console.log(error);
        })
        .finally(() => {
          this.loading = false; // 请求结束后将loading设置为false
        });
    },
    loadFromSession() {
    //   从 session 获取数据
      let metrics = sessionStorage.getItem('complexityMetrics');
      if (metrics) {
        metrics = JSON.parse(metrics);
        for (let key in metrics) {
            if (metrics.hasOwnProperty(key)) {
              let value = metrics[key];
              this.metrics[key] = Math.abs(value);
            }
          }
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