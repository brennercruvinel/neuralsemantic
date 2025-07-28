/**
 * Metrics and Performance Utilities
 */

export class MetricsCollector {
  private metrics: Map<string, any[]> = new Map();
  private startTimes: Map<string, number> = new Map();

  /**
   * Record a metric value
   */
  record(name: string, value: number, tags?: Record<string, string>): void {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }

    this.metrics.get(name)!.push({
      value,
      timestamp: Date.now(),
      tags: tags || {}
    });
  }

  /**
   * Start timing an operation
   */
  startTimer(name: string): void {
    this.startTimes.set(name, Date.now());
  }

  /**
   * End timing and record duration
   */
  endTimer(name: string, tags?: Record<string, string>): number {
    const startTime = this.startTimes.get(name);
    if (!startTime) {
      throw new Error(`Timer '${name}' was not started`);
    }

    const duration = Date.now() - startTime;
    this.record(`${name}_duration_ms`, duration, tags);
    this.startTimes.delete(name);
    
    return duration;
  }

  /**
   * Get metric statistics
   */
  getStats(name: string): { count: number; avg: number; min: number; max: number; latest: number } | null {
    const values = this.metrics.get(name);
    if (!values || values.length === 0) {
      return null;
    }

    const nums = values.map(v => v.value);
    return {
      count: nums.length,
      avg: nums.reduce((a, b) => a + b, 0) / nums.length,
      min: Math.min(...nums),
      max: Math.max(...nums),
      latest: nums[nums.length - 1]
    };
  }

  /**
   * Get all metrics
   */
  getAllMetrics(): Record<string, any> {
    const result: Record<string, any> = {};
    
    for (const [name, values] of this.metrics) {
      result[name] = {
        values: values.slice(-100), // Keep last 100 values
        stats: this.getStats(name)
      };
    }
    
    return result;
  }

  /**
   * Clear all metrics
   */
  clear(): void {
    this.metrics.clear();
    this.startTimes.clear();
  }
}

export class PerformanceProfiler {
  private profiles: Map<string, any> = new Map();
  private activeProfiles: Map<string, { start: number; checkpoints: Array<{ name: string; time: number }> }> = new Map();

  /**
   * Start profiling an operation
   */
  startProfile(name: string): void {
    this.activeProfiles.set(name, {
      start: Date.now(),
      checkpoints: []
    });
  }

  /**
   * Add a checkpoint to active profile
   */
  checkpoint(profileName: string, checkpointName: string): void {
    const profile = this.activeProfiles.get(profileName);
    if (!profile) {
      throw new Error(`Profile '${profileName}' is not active`);
    }

    profile.checkpoints.push({
      name: checkpointName,
      time: Date.now() - profile.start
    });
  }

  /**
   * End profiling and save results
   */
  endProfile(name: string): any {
    const profile = this.activeProfiles.get(name);
    if (!profile) {
      throw new Error(`Profile '${name}' is not active`);
    }

    const totalTime = Date.now() - profile.start;
    const result = {
      name,
      totalTime,
      checkpoints: profile.checkpoints,
      completedAt: new Date().toISOString()
    };

    this.profiles.set(name, result);
    this.activeProfiles.delete(name);
    
    return result;
  }

  /**
   * Get profile results
   */
  getProfile(name: string): any | null {
    return this.profiles.get(name) || null;
  }

  /**
   * Get all profiles
   */
  getAllProfiles(): Record<string, any> {
    const result: Record<string, any> = {};
    for (const [name, profile] of this.profiles) {
      result[name] = profile;
    }
    return result;
  }

  /**
   * Analyze performance bottlenecks
   */
  analyzeBottlenecks(profileName: string): Array<{ stage: string; duration: number; percentage: number }> {
    const profile = this.profiles.get(profileName);
    if (!profile) {
      return [];
    }

    const stages: Array<{ stage: string; duration: number; percentage: number }> = [];
    let lastTime = 0;

    for (const checkpoint of profile.checkpoints) {
      const duration = checkpoint.time - lastTime;
      const percentage = (duration / profile.totalTime) * 100;
      
      stages.push({
        stage: checkpoint.name,
        duration,
        percentage
      });
      
      lastTime = checkpoint.time;
    }

    // Sort by duration descending
    return stages.sort((a, b) => b.duration - a.duration);
  }
}