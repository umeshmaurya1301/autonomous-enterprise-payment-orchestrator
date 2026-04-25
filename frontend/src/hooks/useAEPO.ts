"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { fetchState, postReset, postStep } from "@/lib/api";
import type {
  AEPOObservation,
  AEPOAction,
  ActionLogEntry,
  RewardPoint,
  CausalNotification,
  TaskDifficulty,
  EpisodeStats,
  EpisodeHistoryEntry,
  ObsHistory,
} from "@/lib/types";
import { generateActionLabel } from "@/lib/utils";

const POLL_INTERVAL = 500;
const MAX_REWARD_HISTORY = 100;
const MAX_LOG_ENTRIES = 80;
const MAX_OBS_HISTORY = 25;
const PASS_THRESHOLDS: Record<TaskDifficulty, number> = { easy: 0.75, medium: 0.45, hard: 0.30 };

const OBS_KEYS: (keyof AEPOObservation)[] = [
  "channel", "risk_score", "adversary_threat_level", "system_entropy",
  "kafka_lag", "api_latency", "rolling_p99", "db_connection_pool",
  "bank_api_status", "merchant_tier",
];

function emptyObsHistory(): ObsHistory {
  const h = {} as ObsHistory;
  OBS_KEYS.forEach((k) => { h[k] = []; });
  return h;
}

function checkCausalTransitions(obs: AEPOObservation, prev: AEPOObservation | null): CausalNotification[] {
  const now = Date.now();
  const n: CausalNotification[] = [];
  const t = (id: string, message: string, severity: CausalNotification["severity"]) =>
    n.push({ id: `${id}-${now}`, message, severity, timestamp: now });

  if (obs.kafka_lag > 3000 && (!prev || prev.kafka_lag <= 3000))
    t("kafka", "⚠️ Kafka Lag > 3000 — Latency Compounding", "warning");
  if (obs.kafka_lag > 4000 && (!prev || prev.kafka_lag <= 4000))
    t("kafka-crash", "🔴 Kafka Lag > 4000 — Sim Crash Threshold", "critical");
  if (obs.risk_score > 80 && (!prev || prev.risk_score <= 80))
    t("risk", "⚠️ Risk Score > 80 — Fraud Signal High", "warning");
  if (obs.rolling_p99 > 800 && (!prev || prev.rolling_p99 <= 800))
    t("sla", "🔴 P99 > 800ms — SLA Penalty Active", "critical");
  if (obs.adversary_threat_level > 7 && (!prev || prev.adversary_threat_level <= 7))
    t("adv", "⚠️ Adversary Threat > 7 — Escalation Phase", "warning");
  if (obs.system_entropy > 70 && (!prev || prev.system_entropy <= 70))
    t("entropy", "⚠️ System Entropy > 70 — Latency Spike Imminent", "warning");
  if (obs.db_connection_pool > 80 && (!prev || prev.db_connection_pool <= 80))
    t("db", "⚠️ DB Pool > 80% — Retry Overhead Sim Active", "warning");
  return n;
}

export function useAEPO() {
  const [observation, setObservation] = useState<AEPOObservation | null>(null);
  const [prevObservation, setPrevObservation] = useState<AEPOObservation | null>(null);
  const [obsHistory, setObsHistory] = useState<ObsHistory>(emptyObsHistory());
  const [rewardHistory, setRewardHistory] = useState<RewardPoint[]>([]);
  const [phaseHistory, setPhaseHistory] = useState<string[]>([]);
  const [actionLog, setActionLog] = useState<ActionLogEntry[]>([]);
  const [notifications, setNotifications] = useState<CausalNotification[]>([]);
  const [episodeHistory, setEpisodeHistory] = useState<EpisodeHistoryEntry[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [episodeActive, setEpisodeActive] = useState(false);
  const [episodeDone, setEpisodeDone] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [cumulativeReward, setCumulativeReward] = useState(0);
  const [isResetting, setIsResetting] = useState(false);
  const [lastReward, setLastReward] = useState<number | null>(null);
  const [lastRewardBreakdown, setLastRewardBreakdown] = useState<Record<string, number>>({});
  const [currentTask, setCurrentTask] = useState<TaskDifficulty>("easy");
  const [currentPhase, setCurrentPhase] = useState<string>("—");
  const [curriculumLevel, setCurriculumLevel] = useState<number>(0);
  const [episodeStats, setEpisodeStats] = useState<EpisodeStats | null>(null);
  const [episodeCount, setEpisodeCount] = useState(0);
  const [autoRunInterval, setAutoRunInterval] = useState(600);

  const stepCounterRef = useRef(0);
  const logCounterRef = useRef(0);
  const cumulRef = useRef(0);
  const episodeIdRef = useRef(0);

  const dismissNotification = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const addNotifications = useCallback((notes: CausalNotification[]) => {
    if (!notes.length) return;
    setNotifications((prev) => [...notes, ...prev].slice(0, 6));
    notes.forEach((n) => setTimeout(() => dismissNotification(n.id), 5000));
  }, [dismissNotification]);

  const updateObservation = useCallback((newObs: AEPOObservation) => {
    setObservation((prev) => {
      addNotifications(checkCausalTransitions(newObs, prev));
      setPrevObservation(prev);
      return newObs;
    });
    setObsHistory((hist) => {
      const next = { ...hist };
      OBS_KEYS.forEach((k) => {
        next[k] = [...(hist[k] ?? []).slice(-MAX_OBS_HISTORY + 1), newObs[k] as number];
      });
      return next;
    });
  }, [addNotifications]);

  // Poll /state every 500ms
  useEffect(() => {
    if (!episodeActive) return;
    const id = setInterval(async () => {
      try {
        const result = await fetchState();
        updateObservation(result.observation);
      } catch { /* backend not ready */ }
    }, POLL_INTERVAL);
    return () => clearInterval(id);
  }, [episodeActive, updateObservation]);

  const reset = useCallback(async (task: TaskDifficulty = "easy") => {
    setIsResetting(true);
    setEpisodeDone(false);
    setEpisodeStats(null);
    try {
      const result = await postReset(task);
      updateObservation(result.observation);
      setPrevObservation(null);
      setObsHistory(emptyObsHistory());
      setRewardHistory([]);
      setPhaseHistory([]);
      setActionLog([]);
      setCurrentStep(0);
      setCumulativeReward(0);
      cumulRef.current = 0;
      setLastReward(null);
      setLastRewardBreakdown({});
      setCurrentTask(task);
      setCurrentPhase("normal");
      setCurriculumLevel((result.info?.curriculum_level as number) ?? 0);
      stepCounterRef.current = 0;
      logCounterRef.current = 0;
      episodeIdRef.current += 1;
      setEpisodeActive(true);
      setEpisodeCount((n) => n + 1);
    } catch (err) {
      console.error("Reset failed:", err);
    } finally {
      setIsResetting(false);
    }
  }, [updateObservation]);

  const step = useCallback(async (action: AEPOAction) => {
    if (!episodeActive) return;
    try {
      const result = await postStep(action);
      stepCounterRef.current += 1;
      const stepNum = stepCounterRef.current;

      updateObservation(result.observation);
      setCurrentStep(stepNum);
      setLastReward(result.reward);
      setLastRewardBreakdown(result.reward_breakdown ?? {});

      const phase = (result.info?.phase as string) ?? "normal";
      const currLevel = (result.info?.curriculum_level as number) ?? 0;
      setCurrentPhase(phase);
      setCurriculumLevel(currLevel);
      setPhaseHistory((prev) => [...prev.slice(-MAX_REWARD_HISTORY + 1), phase]);

      cumulRef.current += result.reward;
      const cumul = cumulRef.current;
      setCumulativeReward(cumul);
      setRewardHistory((hist) => [...hist.slice(-MAX_REWARD_HISTORY + 1), { step: stepNum, reward: result.reward, cumulative: cumul }]);

      setActionLog((prev) => [
        {
          id: logCounterRef.current++,
          timestamp: new Date().toLocaleTimeString("en-US", { hour12: false }),
          action,
          reward: result.reward,
          rewardBreakdown: result.reward_breakdown ?? {},
          step: stepNum,
          label: generateActionLabel(action),
          phase,
        },
        ...prev.slice(0, MAX_LOG_ENTRIES - 1),
      ]);

      if (result.done) {
        const avgReward = stepNum > 0 ? cumul / stepNum : 0;
        const passed = avgReward >= PASS_THRESHOLDS[currentTask];
        setEpisodeActive(false);
        setEpisodeDone(true);
        setIsRunning(false);
        const stats: EpisodeStats = { task: currentTask, steps: stepNum, totalReward: cumul, finalPhase: phase, curriculumLevel: currLevel };
        setEpisodeStats(stats);
        setEpisodeHistory((prev) => [
          { id: episodeIdRef.current, task: currentTask, steps: stepNum, avgReward, totalReward: cumul, finalPhase: phase, passed },
          ...prev.slice(0, 19),
        ]);
      }
    } catch (err) {
      console.error("Step failed:", err);
    }
  }, [episodeActive, updateObservation, currentTask]);

  const toggleAutoRun = useCallback(() => setIsRunning((v) => !v), []);
  const dismissDone = useCallback(() => setEpisodeDone(false), []);

  useEffect(() => {
    if (!isRunning || !episodeActive) return;
    const defaultAction: AEPOAction = {
      risk_decision: 0, crypto_verify: 0, infra_routing: 0,
      db_retry_policy: 0, settlement_policy: 0, app_priority: 2,
    };
    const id = setInterval(() => step(defaultAction), autoRunInterval);
    return () => clearInterval(id);
  }, [isRunning, episodeActive, step, autoRunInterval]);

  return {
    observation, prevObservation, obsHistory,
    rewardHistory, phaseHistory,
    actionLog, notifications, episodeHistory,
    isRunning, episodeActive, episodeDone, episodeStats,
    currentStep, cumulativeReward, isResetting,
    lastReward, lastRewardBreakdown,
    currentTask, currentPhase, curriculumLevel, episodeCount,
    autoRunInterval,
    reset, step, toggleAutoRun, dismissNotification, dismissDone,
    setAutoRunInterval,
  };
}
